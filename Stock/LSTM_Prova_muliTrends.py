import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

#Read CSV

# Creating Input Data for LSTM
def create_lstm_data(data, labels, time_steps=1):
    x, y = [], []
    for i in range(len(data) - time_steps):
        x.append(data[i:(i + time_steps)])
        y.append(labels[i])
    return np.array(x), np.array(y)

# Setting Time Steps and Creating Input Data
time_steps = 10
x, y = create_lstm_data(combined_data, labels_scaled, time_steps)
x = np.reshape(x, (x.shape[0], x.shape[1], combined_data.shape[1]))

# Split data into training and testing sets
train_size = int(len(x) * 0.8)
x_train, y_train = x[:train_size], y[:train_size]
x_test, y_test = x[train_size:], y[train_size:]

# Building the LSTM Model for Regression
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_steps, combined_data.shape[1])))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the Model
model.fit(x_train, y_train, epochs=50, batch_size=32)

# Predicting on the Test Data
predictions_scaled = model.predict(x_test)

# Inverse transform the predictions to get back to original scale
predictions = scaler_change.inverse_transform(predictions_scaled)
y_test_inverse = scaler_change.inverse_transform(y_test.reshape(-1, 1))

# Plotting the results
plt.figure(figsize=(14, 5))
plt.plot(data.index[train_size + time_steps + 1: train_size + time_steps + len(y_test) + 1], y_test_inverse, color='blue', label='Actual Change')
plt.plot(data.index[train_size + time_steps + 1: train_size + time_steps + len(predictions) + 1], predictions, color='red', label='Predicted Change')
plt.title('Next Day Stock Price Change Prediction with Google Trends')
plt.xlabel('Date')
plt.ylabel('Price Change ($)')
plt.legend()
plt.show()
