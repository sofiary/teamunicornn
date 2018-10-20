
# coding: utf-8

'''Machne Learning Multi-step forecasts
Forecasting 7 days ahead using half hourly dataset

THis code could be optimize to use multiprocessing with ProcessPoolExecutor
'''

import sys
from math import sqrt
from numpy import split
from numpy import array
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import SGDRegressor


sys.path.append("../scripts/")
from forecast_utils import av_scores

PATH='../input/merged_data/'
MAC_NAME='mac000230'

DAILY_SAMPLE_RATE=48
#one week ahead
FORECAST_DAYS=7
TEST_WEEKS = 1
TEST_SAMPLES=TEST_WEEKS*FORECAST_DAYS*DAILY_SAMPLE_RATE

# split a univariate dataset into train/test sets
def split_dataset(data):
    #use last n weeks for test (enough for validation and test)
    week_hh = FORECAST_DAYS*DAILY_SAMPLE_RATE
    train=data[:-TEST_SAMPLES]
    test = data[-TEST_SAMPLES:]
    # restructure into windows of weekly data
    train = array(split(train, len(train)/week_hh))
    test = array(split(test, len(test)/week_hh))
    return train, test

# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
    print('>evaluate_forecasts()')
    scores = list()
    # calculate an RMSE score for each day
    for i in range(actual.shape[1]):
        # calculate mse
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        # calculate rmse
        rmse = sqrt(mse)
        # store
        scores.append(rmse)
    # calculate overall RMSE
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col])**2
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores

# summarize scores
def print_scores(name, score, scores):
    #s_scores = ', '.join(['%.1f' % s for s in scores])
    #print('%s: [%.3f] %s' % ('Half hourly', score, s_scores))
    n_chunks = len(scores)/DAILY_SAMPLE_RATE
    print(type(scores))
    scores_chunked = np.array_split(scores, n_chunks)
    av_scores = []
    for chunk in scores_chunked:
        av_scores.append(np.average(chunk))
    w_scores = ', '.join(['%.1f' % s for s in av_scores])
    print('%s: [%.3f] %s' % (name, score, w_scores))

def print_best_algo(name_score):
    best_alg = ''
    best_score = 0
    for i, n_s in enumerate(name_score):
        if i == 0:
            best_alg=n_s[0]
            best_score=n_s[1]
        else:
            if n_s[1]<best_score:
                best_alg=n_s[0]
                best_score=n_s[1]
    print('Best overall algorithm: {0}'.format(best_alg))

# prepare a list of ml models
def get_models(models=dict()):
    # linear models
    models['linear regression'] = LinearRegression()
    models['lasso'] = Lasso()
    models['ridge'] = Ridge()
    models['elastic net'] = ElasticNet()
    models['huber regressor'] = HuberRegressor()
    #models['lars'] = Lars()
    models['lasso lars'] = LassoLars()
    models['passive aggressive regressor'] = PassiveAggressiveRegressor(max_iter=1000, tol=1e-3)
    models['ranscac regressor'] = RANSACRegressor(min_samples=7)
    models['sgd regressor'] = SGDRegressor(max_iter=5000, tol=1e-3)
    print('Defined %d models' % len(models))
    return models

# create a feature preparation pipeline for a model
def make_pipeline(model):
    steps = list()
    # standardization
    steps.append(('standardize', StandardScaler()))
    # normalization
    steps.append(('normalize', MinMaxScaler()))
    # the model
    steps.append(('model', model))
    # create pipeline
    pipeline = Pipeline(steps=steps)
    return pipeline

# convert history into inputs and outputs
def to_supervised(history, output_ix, data_column):
    X, y = list(), list()
    # step over the entire history one time step at a time
    for i in range(len(history)-1):
        for j in range(DAILY_SAMPLE_RATE):
            #as we are sub-sampling individual days by the DAILY_SAMPLE_RATE
            #we need to use part of the next week to predict the current step
            rolling_window = history[i][j:,data_column]
            rolling_window_next = history[i+ 1][:j, data_column]
            rolling_window = np.append(rolling_window, rolling_window_next)
            X.append(rolling_window)
            y.append(history[i + 1][output_ix,data_column])
    assert(len(X)==len(y))
    assert(len(X[0])==DAILY_SAMPLE_RATE*FORECAST_DAYS)
    return array(X), array(y)

# fit a model and make a forecast
def sklearn_predict(model, history, data_column):
    yhat_sequence = list()
    # fit a model for each step
    for i in range(FORECAST_DAYS*DAILY_SAMPLE_RATE):
        # prepare data
        train_x, train_y = to_supervised(history, i, data_column)
        # make pipeline
        pipeline = make_pipeline(model)
        # fit the model
        pipeline.fit(train_x, train_y)
        # forecast
        x_input = array(train_x[-1, :]).reshape(1,FORECAST_DAYS*DAILY_SAMPLE_RATE)
        yhat = pipeline.predict(x_input)[0]
        # store
        yhat_sequence.append(yhat)
    return yhat_sequence

# evaluate a single model
def evaluate_model(model, train, test, data_column=0):
    '''By default use first column as the data column we train and predict for'''
    # history is a list of weekly data
    history = [x for x in train]
    # walk-forward validation over each week
    predictions = list()
    for i in range(len(test)):
        # predict the week
        yhat_sequence = sklearn_predict(model, history, data_column)
        # store the predictions
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next week
        history.append(test[i, :])
    predictions = array(predictions)
    # evaluate predictions days for each week
    actuals = test[:, :, data_column]
    score, scores = evaluate_forecasts(actuals, predictions)
    return score, scores, actuals, predictions

def eval_model(name, model):
    print('>eval_model() {0}'.format(name))
    # evaluate and get scores, energy(kWh/hh) is the first column in our dataset
    score, scores, actuals, predictions = evaluate_model(model, train, test, data_column=0)
    # summarize scores
    print_scores(name, score, scores)
    av = av_scores(scores, DAILY_SAMPLE_RATE)
    # plot scores
    return av, actuals, predictions

# ### Read in data
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

dataset = pd.read_csv('{0}LCLid/clean_{1}.csv'.format(PATH, MAC_NAME), parse_dates=['day_time'], date_parser=dateparse)

#subsample for testing
#dataset = dataset[-1*(4*DAILY_SAMPLE_RATE*FORECAST_DAYS):]

dataset.set_index(['day_time'],inplace=True)
#original data is to 3 decimal places
dataset['energy(kWh/hh)'] = dataset['energy(kWh/hh)'].round(3)
dataset['energy(Wh/hh)'] = dataset['energy(kWh/hh)'].multiply(1000)

dataset =  dataset[['energy(Wh/hh)', 'temperature', 'humidity']]
dataset=dataset.fillna(0)

ts = dataset['energy(Wh/hh)'].values
ds = dataset.index.values

plt.plot(ds, ts)
plt.show(block=False)
plt.close()

# split into train and test
train, test = split_dataset(dataset.values)
# prepare the models to evaluate
models = get_models()

days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
score_list = []


avs=[]
actuals=[]
predictions=[]
names=[]
for name, model in models.items():
    av, actual, prediction = eval_model(name, model)
    names.append(name)
    avs.append(av)
    score_list.append((name, av))
    predictions.append(prediction)
    actuals.append(actual)

for name, av in zip(names, avs):
    plt.scatter(days, av, marker='o', label=name)
    plt.ylabel('RSWE Wh')
# show plot
plt.legend()
plt.savefig('plots/4_2_a_{0}_{1}_day_forecast_rmse.png'.format(MAC_NAME, FORECAST_DAYS))
plt.show(block=False)

i=0
hh=range(FORECAST_DAYS*DAILY_SAMPLE_RATE)
for name, actual, pred in zip(names, actuals, predictions):
    plt.subplot(len(models.items()), 1, i+1)
    actual=actual.flatten()
    plt.plot(hh, actual, label='actual', color='k')
    prediction = pred.flatten()
    plt.plot(hh, prediction,  label='predicted', color='r')
    plt.title(name, fontsize=10)
    plt.ylabel('Wh/hh')
    i+=1

plt.legend()
plt.savefig('plots/4_2_a_{0}_{1}_day_forecast_actuals.png'.format(MAC_NAME, FORECAST_DAYS))
plt.show(block=False)



best_alg = print_best_algo(score_list)

# Display history and forecasts on same plot

# Run models on daily dataset
