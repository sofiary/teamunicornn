# recursive multi-step forecast with linear algorithms
import time
import sys
from math import sqrt
import numpy as np
import pickle as pkl
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

PATH='../input/merged_data/LCLid/clean/'
MODEL_NAME = '4_2_b_ml_recursive'

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
    train = np.array(np.split(train, len(train)/week_hh))
    test = np.array(np.split(test, len(test)/week_hh))
    return train, test


# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
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
            s += (actual[row, col] - predicted[row, col]) ** 2
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores

# summarize scores
def print_scores(name, score, scores):
    #s_scores = ', '.join(['%.1f' % s for s in scores])
    #print('%s: [%.3f] %s' % ('Half hourly', score, s_scores))
    n_chunks = len(scores)/DAILY_SAMPLE_RATE
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


# make a recursive multi-step forecast
# make a prediction for one time step, taking the prediction, feed it into the model as an input in order to predict the subsequent time step. Repeat until the desired number of steps have been forecast.
def forecast(model, input_x, n_input):
    yhat_sequence = list()
    input_data = [x for x in input_x]
    for j in range(FORECAST_DAYS*DAILY_SAMPLE_RATE):
        # prepare the input data
        X = np.array(input_data[-n_input:]).reshape(1, n_input)
        # make a one-step forecast
        yhat = model.predict(X)[0]
        # add to the result
        yhat_sequence.append(yhat)
        # add the prediction to the input
        input_data.append(yhat)
    return yhat_sequence


# convert windows of weekly multivariate data into a series of total power
def to_series(data):
    # extract just the total power from each week
    series = [week[:, 0] for week in data]
    # flatten into a single series
    series = np.array(series).flatten()
    return series


# convert history into inputs and outputs
def to_supervised(history, n_input):
    # convert history to a univariate series
    data = to_series(history)
    X, y = list(), list()
    ix_start = 0
    # step over the entire history one time step at a time
    for i in range(len(data)):
        # define the end of the input sequence
        ix_end = ix_start + n_input
        # ensure we have enough data for this instance
        if ix_end < len(data):
            X.append(data[ix_start:ix_end])
            y.append(data[ix_end])
        # move along one time step
        ix_start += 1
    return np.array(X), np.array(y)


# fit a model and make a forecast
def sklearn_predict(model, history, n_input):
    # prepare data
    train_x, train_y = to_supervised(history, n_input)
    # make pipeline
    pipeline = make_pipeline(model)
    # fit the model
    pipeline.fit(train_x, train_y)
    # predict the week, recursively
    yhat_sequence = forecast(pipeline, train_x[-1, :], n_input)
    return yhat_sequence


# evaluate a single model
def evaluate_model(model, train, test, n_input, data_column=0):
    # history is a list of weekly data
    history = [x for x in train]
    # walk-forward validation over each week
    predictions = list()
    for i in range(len(test)):
        # predict the week
        yhat_sequence = sklearn_predict(model, history, n_input)
        # store the predictions
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next week
        history.append(test[i, :])
    predictions = np.array(predictions)
    # evaluate predictions days for each week
    actuals = test[:, :, data_column]
    score, scores = evaluate_forecasts(actuals, predictions)
    return score, scores, actuals, predictions

def av_scores(scores, sample_rate=48):
    n_chunks = len(scores) / sample_rate
    scores_chunked = np.array_split(scores, n_chunks)
    av_scores = []
    for chunk in scores_chunked:
        av_scores.append(np.mean(chunk))
    return av_scores

def eval_model(name, model, train, test, n_input, data_column=0):
    # evaluate and get scores, energy(kWh/hh) is the first column in our dataset
    score, scores, actuals, predictions = evaluate_model(model, train, test, n_input, data_column)
    # summarize scores
    print_scores(name, score, scores)
    av = av_scores(scores, DAILY_SAMPLE_RATE)
    # plot scores
    return av, actuals, predictions

def create_dataset(mac, data_col = ['energy(kWh/hh)'], train_cols=['temperature', 'humidity']):
    # ### Read in data
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

    dataset = pd.read_csv('{0}{1}.csv'.format(PATH, mac), parse_dates=['day_time'], date_parser=dateparse)

    #subsample for testing
    #dataset = dataset[-1*(4*DAILY_SAMPLE_RATE*FORECAST_DAYS):]

    dataset.set_index(['day_time'],inplace=True)
    #original data is to 3 decimal places
    dataset[data_col] = dataset[data_col].round(3)
    #dataset['energy(Wh/hh)'] = dataset['energy(kWh/hh)'].multiply(1000)

    dataset = dataset[data_col + train_cols]
    dataset=dataset.fillna(0)
    return dataset

def save_obj(obj, path, name ):
    with open(path + name + '.pkl', 'wb') as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)

def load_obj(path, name ):
    with open(path + name + '.pkl', 'rb') as f:
        return pkl.load(f)

def save_eval_pkl(av, scores, name):
    save_obj(av, '../results/', 'forecast_averages_{0}'.format(name))
    save_obj(scores, '../results/', 'forecast_scores_{0}'.format(name))


def run_forecast(dataset, mac_name, model_name):

    # split into train and test
    train, test = split_dataset(dataset.values)
    # prepare the models to evaluate
    models = get_models()

    n_input = FORECAST_DAYS*DAILY_SAMPLE_RATE
    # evaluate each model
    days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
    avs=[]
    actuals=[]
    predictions=[]
    names=[]
    score_list = []
    for name, model in models.items():
        # evaluate and get scores
        av, actual, prediction = eval_model(name, model, train, test, n_input, data_column=0)
        names.append(name)
        avs.append(av)
        score_list.append((name, av))
        predictions.append(prediction)
        actuals.append(actual)

    for name, av in zip(names, avs):
        plt.plot(days, av, marker='o', label=name)
        plt.ylabel('RSWE Wh')
    # show plot
    plt.legend()
    #plt.savefig('plots/4_2_b_{0}_{1}_day_forecast_rmse.png'.format(mac, FORECAST_DAYS))
    name_fig = 'plots/{0}_{1}'.format(model_name, mac_name)
    plt.savefig(name_fig)
    plt.show(block=False)
    plt.close()
    best_alg = print_best_algo(score_list)
    return names, actuals, predictions, models

def plot_forecasts(names, actuals, predictions, models, mac):
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
    plt.savefig('plots/{0}_{1}_{2}_day_forecast_actuals.png'.format(MODEL_NAME, mac, FORECAST_DAYS))
    plt.show(block=False)


def workflow(maclist = ['mac000230', 'mac000100'], data_col = ['energy(kWh/hh)'], train_cols=['temperature', 'humidity']):
    start = time.time()
    for mac in maclist:
        dataset = None
        dataset = create_dataset(mac, data_col, train_cols)
        names, actuals, predictions, models=run_forecast(dataset, mac, MODEL_NAME)
        plot_forecasts(names, actuals, predictions, models, mac)
    end = time.time()
    elapsed = end - start
    print('<<workflow() for {0} macs took {1} secs'.format(len(maclist), elapsed))

if __name__ == "__main__":
    workflow()
