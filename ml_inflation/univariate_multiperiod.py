"""
First attempt, univariate-multperiod try.  Predicts the next 12 months of core inflation
from just the core inflation series
"""
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from ml_inflation.data import InflationSeries


def get_path():
    """
    get the file path for the model output
    :return:
    """
    path = os.path.join(os.getcwd(), 'modelout')
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def timeseries_to_supervised(data, lag=1):
    """
    frame a sequence as a supervised learning problem
    :param data: the time series data
    :param lag: how long to lag it
    :return:
    """
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


def difference(dataset, interval=1):
    """
    create a differenced series
    :param dataset: the time series
    :param interval: the length of the difference
    :return: the differenced time series
    """
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pd.Series(diff)


def inverse_difference(history, yhat, interval=1):
    """
    invert differenced value
    :param history: the data to invert
    :param yhat: the forecast to invert
    :param interval: the difference interval
    :return:
    """
    return yhat + history[-interval]


def scale(train, test):
    """
    scale train and test data to [-1, 1]
    :param train: training data
    :param test: test data
    :return: the scaler, the train_scaled, the test_scaled
    """
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


def invert_scale(scaler, X, value):
    """
    inverse scaling for a forecasted value
    :param scaler: the scaler
    :param X: the forecasts
    :param value: the last value
    :return: inverted forecasts
    """
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


def fit_lstm(train, batch_size, nb_epoch, neurons, ):
    """
    Create and fit an LTSM model
    :param train: the training data
    :param batch_size: the batch size
    :param nb_epoch: the number of epocks
    :param neurons: the number of nurons
    :return: a trained LTSM model
    """
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]),
                   stateful=True, return_sequences=True))
    model.add(LSTM(neurons, stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
        if (i + 1) % 10 == 0:
            print("Run {} of {}".format(i + 1, nb_epoch))
    return model


def save_ltsm_model(model):
    """
    Save a trained model to disk
    :param model: the model
    """
    savepath = os.path.join(get_path(), 'uvop.h5f')
    print('Saving model to {}'.format(savepath))
    model.save(savepath)


def load_ltsm_model():
    """
    Load a saved model from disk
    :return: the model
    """
    filepath = os.path.join(get_path(), 'uvop.h5f')
    print('Loading model from {}'.format(filepath))
    model = load_model(filepath)
    return model


def forecast_lstm(model, batch_size, X):
    """
    make a one-step forecast
    :param model: the model
    :param batch_size: the batch size
    :param X: the X to generate predictions from
    :return: the predicted Y
    """
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]


# load dataset
series = InflationSeries.get_core().get_series()
# transform data to be stationary
raw_values = series.values
diff_values = difference(raw_values, 1)

# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values

# split data into train and test-sets
train, test = supervised_values[0:-12], supervised_values[-12:]

# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)

# fit the model
loadModel = False

if loadModel:
    model = load_ltsm_model()
else:
    model = fit_lstm(train, 1, 1000, 128)

if not loadModel:
    save_ltsm_model(model)

# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
model.predict(train_reshaped, batch_size=1)

# walk-forward validation on the test data
predictions = list()
for i in range(len(test_scaled)):
    # make one-step forecast
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = forecast_lstm(model, 1, X)
    # invert scaling
    yhat = invert_scale(scaler, X, yhat)
    # invert differencing
    yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
    # store forecast
    predictions.append(yhat)
    expected = raw_values[len(train) + i + 1]
    err = np.divide(yhat, expected) - 1
    print('Month={0}, Predicted={1:.3f}, Expected={2:.3f}, Diff={3:.3f}'.format(i + 1,
                                                                                yhat,
                                                                                expected,
                                                                                err * 100))

# report performance
rmse = math.sqrt(mean_squared_error(raw_values[-12:], predictions))
print('Test RMSE: %.3f' % rmse)
# line plot of observed vs predicted
fig, ax = plt.subplots(figsize=(12, 9))
ax.plot(raw_values[-12:], label='Observed')
ax.plot(predictions, label='Predicted')
ax.legend()
plt.show()

mom_pred = (pd.Series(predictions).pct_change(1) * 100).dropna()
mom_act = (pd.Series(raw_values[-12:]).pct_change(1) * 100).dropna()

fig, ax = plt.subplots(figsize=(12, 9))
ax.plot(mom_pred, label='Predicted')
ax.plot(mom_act, label='Actual')
ax.legend()
plt.show()

for p, a in zip(mom_pred, mom_act):
    print('Predicted={:.3f}, Actual={:.3f}'.format(p, a))
