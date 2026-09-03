import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# ----------------------------------------
# PREPARE TIME SERIES DATA
# ----------------------------------------

def prepare_lstm_data(data, time_step=10):

    X = []
    y = []

    for i in range(len(data) - time_step - 1):

        X.append(data[i:(i + time_step), 0])

        y.append(data[i + time_step, 0])

    return np.array(X), np.array(y)

# ----------------------------------------
# TRAIN LSTM MODEL
# ----------------------------------------

def train_lstm(df, target_column):

    dataset = df[[target_column]].values

    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_data = scaler.fit_transform(dataset)

    # PREPARE DATA

    X, y = prepare_lstm_data(scaled_data)

    X = X.reshape(X.shape[0], X.shape[1], 1)

    # BUILD MODEL

    model = Sequential()

    model.add(
        LSTM(
            50,
            return_sequences=True,
            input_shape=(X.shape[1], 1)
        )
    )

    model.add(LSTM(50))

    model.add(Dense(1))

    model.compile(
        loss='mean_squared_error',
        optimizer='adam'
    )

    # TRAIN

    model.fit(
        X,
        y,
        epochs=5,
        batch_size=16,
        verbose=0
    )

    # PREDICTIONS

    predictions = model.predict(X)

    predictions = scaler.inverse_transform(predictions)

    actual = scaler.inverse_transform(y.reshape(-1, 1))

    return actual, predictions