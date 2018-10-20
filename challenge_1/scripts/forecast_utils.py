from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from math import sqrt
from numpy import split
from numpy import array
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd


# split a univariate dataset into train/test sets
def split_dataset(data, sample_rate=48):
    # split into weeks 
    #use last 8 weeks for test (enough for validation and test)
    week_hh = 7*sample_rate
    test_days = 8*week_hh
    train, test = data[:-(test_days)], data[test_days:]
    print('train: {0}, split weeks: {1}'.format(len(train), len(train)/week_hh))
    print('test: {0}, split weeks: {1}'.format(len(test), len(test)/week_hh))
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
        #all rows : and ith column
        #prdicted is length 7, ie 0 to 6 indexes
        # prdicted is length 336
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
def summarize_scores(name, score, scores, sample_rate=48):
    n_chunks = len(scores)/sample_rate
    print(type(scores))
    scores_chunked = np.array_split(scores, n_chunks)
    av_scores = []
    for chunk in scores_chunked:
        av_scores.append(np.average(chunk))
    w_scores = ', '.join(['%.1f' % s for s in av_scores])
    print('%s: [%.3f] %s' % (name, score, w_scores))
 
# evaluate a single model
def evaluate_model(model_func, train, test):
    # history is a list of weekly data
    history = [x for x in train]
    # walk-forward validation over each week
    predictions = list()
    for i in range(len(test)):
        # predict the week
        yhat_sequence = model_func(history)
        # store the predictions
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next week
        history.append(test[i, :])
    predictions = array(predictions)
    # evaluate predictions days for each week
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    return score, scores

def av_scores(scores, sample_rate=48):
    n_chunks = len(scores) / sample_rate
    scores_chunked = np.array_split(scores, n_chunks)
    av_scores = []
    for chunk in scores_chunked:
        av_scores.append(np.mean(chunk))
    return av_scores

