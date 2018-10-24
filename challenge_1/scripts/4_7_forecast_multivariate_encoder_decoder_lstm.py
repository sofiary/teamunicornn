# multivariate multi-step encoder-decoder lstm
import time
from math import sqrt
from numpy import split
from numpy import array
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

PATH='../input/merged_data/LCLid/clean/'

MODEL_NAME = '4_7_mutivariate_enc_dec_lstm'
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

# convert history into inputs and outputs
def to_supervised(train, n_input, n_out=7):
    # flatten data
    data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
    X, y = list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end < len(data):
            X.append(data[in_start:in_end, :])
            y.append(data[in_end:out_end, 0])
        # move along one time step
        in_start += 1
    return array(X), array(y)

# train the model
def build_model(train, n_input, n_out):
    # prepare data
    train_x, train_y = to_supervised(train, n_input, n_out)
    # define parameters, try with 50 epochs vs 10
    verbose, epochs, batch_size = 0, 10, 16
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # reshape output into [samples, timesteps, features]
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    # define model
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer='adam')
    # fit network
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model

# make a forecast
def forecast(model, history, n_input):
    # flatten data
    data = array(history)
    data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    # retrieve last observations for input data
    input_x = data[-n_input:, :]
    # reshape into [1, n_input, n]
    input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
    # forecast the next week
    yhat = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]
    return yhat

# evaluate a single model
def evaluate_model(train, test, n_input, n_out):
    # fit model
    model = build_model(train, n_input, n_out)
    # history is a list of weekly data
    history = [x for x in train]
    # walk-forward validation over each week
    predictions = list()
    for i in range(len(test)):
        # predict the week
        yhat_sequence = forecast(model, history, n_input)
        # store the predictions
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next week
        history.append(test[i, :])
    # evaluate predictions days for each week
    predictions = array(predictions)
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    return score, scores

def av_scores(scores, sample_rate=48):
    n_chunks = len(scores) / sample_rate
    scores_chunked = np.array_split(scores, n_chunks)
    av_scores = []
    for chunk in scores_chunked:
        av_scores.append(np.mean(chunk))
    return av_scores

def save_obj(obj, path, name ):
    with open(path + name + '.pkl', 'wb') as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)

def load_obj(path, name ):
    with open(path + name + '.pkl', 'rb') as f:
        return pkl.load(f)

def save_eval_pkl(av, scores, name):
    save_obj(av, '../results/', 'forecast_averages_{0}'.format(name))
    save_obj(scores, '../results/', 'forecast_scores_{0}'.format(name))

def run_forecast(dataset, name, model_name):
    # split into train and test
    train, test = split_dataset(dataset.values)

    # evaluate model and get scores
    n_input = FORECAST_DAYS * DAILY_SAMPLE_RATE
    n_out = FORECAST_DAYS * DAILY_SAMPLE_RATE
    score, scores = evaluate_model(train, test, n_input, n_out)

    # summarize scores
    print_scores('cnn', score, scores)
    av = av_scores(scores, DAILY_SAMPLE_RATE)

    # plot scores
    days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
    plt.plot(days, av, marker='o', label='lstm')
    plt.savefig('plots/{0}_{1}'.format( model_name, name))
    save_eval_pkl(av, scores, name+'_'+model_name)
    #plt.show()

def create_dataset(mac, data_col = ['energy(kWh/hh)'], train_cols=['temperature', 'humidity'], is_test=False):
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

    dataset = pd.read_csv('{0}{1}.csv'.format(PATH, mac), parse_dates=['day_time'],
                          date_parser=dateparse)
    if is_test:
        # subsample for testing
        dataset = dataset[-1 * (8 * DAILY_SAMPLE_RATE * FORECAST_DAYS):]
        print('is_test==True')

    dataset.set_index(['day_time'], inplace=True)
    # original data is to 3 decimal places
    dataset[data_col] = dataset[data_col].round(3)
    #dataset['energy(Wh/hh)'] = dataset['energy(kWh/hh)'].multiply(1000)
    #dataset = dataset[['energy(Wh/hh)', 'temperature', 'humidity']]

    #all other models we are using predict kWh so lets keep consistent
    dataset = dataset[data_col + train_cols]
    dataset = dataset.fillna(0)
    return dataset


def workflow(maclist = ['mac000230','mac000100'], data_col = ['energy(kWh/hh)'], train_cols=['temperature', 'humidity'], is_test=False):
    start = time.time()
    for mac in maclist:
        dataset = create_dataset(mac,data_col, train_cols, is_test)
        run_forecast(dataset, mac, MODEL_NAME)
    end = time.time()
    elapsed = end - start
    print('<<workflow() for {0} macs took {1} secs'.format(len(maclist), elapsed))

if __name__ == "__main__":
    workflow()