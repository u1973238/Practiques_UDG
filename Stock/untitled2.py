import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Downloading Historical Stock Data
stock_symbol = 'AAPL'
start_date = '2020-01-01'
end_date = '2021-01-01'
data = yf.download(stock_symbol, start=start_date, end=end_date)

# Data Preprocessing
close_prices = data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
close_prices_scaled = scaler.fit_transform(close_prices)

# Creating Input Data for LSTM
def create_lstm_data(data, time_steps=1):
    x, y = [], []
    for i in range(len(data) - time_steps):
        x.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(x), np.array(y)

# Setting Time Steps and Creating Input Data
time_steps = 10
x, y = create_lstm_data(close_prices_scaled, time_steps)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# Building the LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the Model
model.fit(x, y, epochs=50, batch_size=32)

# Predicting Future Stock Prices
future_days = 30
predicted_prices = []

# Use the last time_steps data for predicting future prices
last_data = close_prices_scaled[-time_steps:].reshape(1, time_steps, 1)

for _ in range(future_days):
    pred = model.predict(last_data)
    predicted_prices.append(pred[0, 0])
    last_data = np.append(last_data[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

# Inverse transform the predicted prices
predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

# Displaying Predictions
future_dates = pd.date_range(start=end_date, periods=future_days + 1)[1:]  # Skip the start date
future_data = pd.DataFrame({'Date': future_dates, 'Predicted Price': predicted_prices.flatten()})
print(future_data)

# Plotting the results
plt.figure(figsize=(14, 5))
plt.plot(data['Close'], label='Original Data')
plt.plot(pd.date_range(start=data.index[-1], periods=future_days + 1)[1:], predicted_prices, label='Predicted Data')
plt.legend()
plt.show()
