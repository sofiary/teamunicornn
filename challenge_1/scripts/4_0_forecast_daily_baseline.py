
# coding: utf-8

# ### Baseline forecast for total daily consumption per household

# Select n households from the total
# 
# Forecast total consumption for each household using daily totals as dataset
# 
# 'datascience' virtualenv (tensorflow, sklearn pandas etc...)

import time
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from math import sqrt
from numpy import split
from numpy import array
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import pandas as pd
import dask
import dask.dataframe as dd
import seaborn as sns; sns.set()

PATH='../input/merged_data/LCLid/clean/'
MODEL_NAME = '4_0_daily_baseline'

#half hourly data
DAILY_SAMPLE_RATE = 48
FORECAST_DAYS=7
TEST_WEEKS = 1
TEST_SAMPLES=TEST_WEEKS*FORECAST_DAYS*DAILY_SAMPLE_RATE

pd.set_option('display.max_columns', None)

#disable SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'

# split a univariate dataset into train/test sets
def split_dataset(data):
    # split into weeks 
    #use last 8 weeks for test (enough for validation and test)
    week_hh = FORECAST_DAYS*DAILY_SAMPLE_RATE
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
    #print('actual.shape: {0}'.format(actual.shape))
    #print('actual.shape[1]: {0}'.format(actual.shape[1]))
    #print('predicted.shape[1]: {0}'.format(predicted.shape[1]))
    #print('first row all col. actual [0, :]: {0}'.format(actual[0, :]))

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
def summarize_scores(name, score, scores):
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
 
# evaluate a single model
def evaluate_model(model_func, train, test):
    # history is a list of weekly data
    history = [x for x in train]
    # walk-forward validation over each week
    #print('test: {0}, train: {1}'.format(len(test), len(train)))
    #print('history[0]: {0}'.format(history[0]))
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

# daily persistence model
def daily_persistence(history):
    # get the data for the prior week
    #print('len(history): {0}'.format(len(history)))
    last_week = history[-1]
    # get the total value for the last day
    value = last_week[-1, 0]
    # prepare 7 day forecast
    forecast = [value for _ in range(7*DAILY_SAMPLE_RATE)]
    return forecast
 
# weekly persistence model
def weekly_persistence(history):
    # get the data for the prior week
    last_week = history[-1]
    return last_week[:, 0]
 
# week one year ago persistence model
def week_one_year_ago_persistence(history):
    # get the data for the prior week
    last_week = history[-52]
    return last_week[:, 0]

def av_scores(scores):
    n_chunks = len(scores) / DAILY_SAMPLE_RATE
    scores_chunked = np.array_split(scores, n_chunks)
    av_scores = []
    for chunk in scores_chunked:
        av_scores.append(np.average(chunk))
    return av_scores

def run_forecast(dataset, name, model_name):

    # split into train and test
    train, test = split_dataset(dataset.values)

    # define the names and functions for the models we wish to evaluate
    models = dict()
    models['daily'] = daily_persistence
    models['weekly'] = weekly_persistence
    models['week-one_year_ago'] = week_one_year_ago_persistence

    # evaluate each model
    days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
    # pseododays = list(range(1,8))
    for name, func in models.items():
        # evaluate and get scores
        score, scores = evaluate_model(func, train, test)
        # summarize scores
        summarize_scores(name, score, scores)
        # plot scores
        print('type(scores): {0}'.format(type(scores)))
        av = av_scores(scores)
        print('av_scores: {0}, len: {1}, type: {2}'.format(av, len(av), type(av), type(av[0])))
        print('days: {0}, type: {1}'.format(days, type(days)))
        # scores=[567.6262062743326, 500.34052595963686, 411.23549302262967, 466.10635038191396, 471.884456004443,
        #         358.3000946061587, 481.9964533953047]
        sns.scatterplot(days, av, marker='o', label=name)
        # plt.scatter(days, scores, marker='o', label=name)
    # show plot
    plt.legend()
    plt.savefig('plots/{0}_{1}'.format(model_name, name))
    plt.show(block=False)

def infill_empty(start, end, mac):
    # Check dates
    # First check for any gaps in the half hourly data
    print(mac['energy(kWh/hh)'].first_valid_index())
    # find the last valid value
    print(mac['energy(kWh/hh)'].last_valid_index())
    # Lets start on the first Sunday morning after first valid data
    # And end on Last Sunday morning before last valid point

    delta = relativedelta(end, start)

    # Clip the data to the required range
    mac['day'] = mac.index
    mask = (mac['day'] >= start) & (mac['day'] < end)
    mac = mac.loc[mask]

    # Note date_range is inclusive of the end date
    # Test date range will give us correct values for a single day
    test_date_range = pd.date_range('2011-12-4 00:00:00', '2011-12-4 23:30:00', freq='30Min')
    print(len(test_date_range) / DAILY_SAMPLE_RATE)
    test_date_range[0], test_date_range[-1]

    ref_date_range = pd.date_range('2011-12-4 00:00:00', '2014-2-22 23:30:00', freq='30Min')

    ref_df = pd.DataFrame(np.random.randint(1, 20, (ref_date_range.shape[0], 1)))
    ref_df.index = ref_date_range  # set index

    # check for missing datetimeindex values based on reference index (with all values)
    missing_dates = ref_df.index[~ref_df.index.isin(mac.index)]

    # TODO some columns need to be adjusted (eg sum)

    prev_row = mac.loc['2013-09-09 22:30:00'].copy()
    post_row = mac.loc['2013-09-10 01:00:00'].copy()

    # As a quick hack we are just going to duplicate the prev row 2x and post row 2x to fill in the gaps

    mac.loc[pd.to_datetime('2013-09-09 23:00:00')] = prev_row
    mac.loc[pd.to_datetime('2013-09-09 23:30:00')] = prev_row
    mac.loc[pd.to_datetime('2013-09-10 00:00:00')] = post_row
    mac.loc[pd.to_datetime('2013-09-10 00:30:00')] = post_row

    mac = mac.sort_index()

    # double check missing dates were fixed
    missing_dates = ref_df.index[~ref_df.index.isin(mac.index)]

    print(
        'frac missing: {0}, total: {1}, total days: {2}'.format(len(missing_dates) / len(mac), len(mac), len(mac) / DAILY_SAMPLE_RATE))

    # check if more than necessary in data
    inv_missing_dates = mac.index[~mac.index.isin(ref_df.index)]
    print('frac missing: {0}, inv_missing_dates: {1}'.format(len(inv_missing_dates) / len(mac), inv_missing_dates))
    #mac.to_csv('{0}LCLid/clean_mac000230.csv'.format(PATH))
    return mac


def create_dataset(name):
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

    mac = pd.read_csv('{0}{1}.csv'.format(PATH,name), parse_dates=['day_time'], date_parser=dateparse)

    mac.sort_values(by=['day_time'], inplace=True)
    mac.set_index(['day_time'], inplace=True)

    mac_df = mac[['energy(kWh/hh)', 'temperature', 'humidity']]
    mac_df = mac_df.fillna(0)
    return mac_df


def workflow():
    wflow_start = time.time()
    # start at start of Sunday
    start = datetime(2011, 12, 4)
    # End at start of Sunday
    end = datetime(2014, 2, 23)
    maclist = ['mac000230', 'mac000100']
    for mac in maclist:
        dataset = create_dataset(mac)
        #dataset = infill_empty(start, end, dataset)
        run_forecast(dataset, mac, MODEL_NAME)
    wflow_end = time.time()
    elapsed = wflow_end - wflow_start
    print('<<workflow() for {0} macs took {1} secs'.format(len(maclist), elapsed))

if __name__ == "__main__":
    workflow()