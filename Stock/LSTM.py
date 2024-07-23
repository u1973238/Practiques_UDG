import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Downloading Historical Stock Data
stock_symbol = 'AAPL'
start_date = '2023-07-01'
end_date = '2024-07-01'
data = yf.download(stock_symbol, start=start_date, end=end_date)

# Creating the target column
data['target'] = data['Close'].shift(-1) - data['Close']

# Data Preprocessing
close_prices = data['target'].values.reshape(-1, 1)
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

# Splitting the data into training and test sets
train_size = int(len(x) * 0.8)
test_size = len(x) - train_size
x_train, x_test = x[0:train_size], x[train_size:len(x)]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

# Building the LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_steps, 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the Model
model.fit(x_train, y_train, epochs=50, batch_size=32)

# Predicting on the Test Data
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Predicting the next day's price
last_time_step_data = close_prices_scaled[-time_steps:]
last_time_step_data = np.reshape(last_time_step_data, (1, time_steps, 1))

next_day_prediction_scaled = model.predict(last_time_step_data)
next_day_prediction = scaler.inverse_transform(next_day_prediction_scaled)

# Plotting the results
plt.figure(figsize=(14, 5))
plt.plot(data.index[train_size + time_steps:], y_test, color='blue', label='Actual Prices')
plt.plot(data.index[train_size + time_steps:], predictions, color='red', label='Predicted Prices')

# Extend the plot to include the next day's prediction
extended_dates = np.append(data.index, data.index[-1] + pd.Timedelta(days=1))
extended_actual_prices = np.append(close_prices, np.nan)
extended_predictions = np.append(predictions.flatten(), next_day_prediction[0][0])

plt.plot(extended_dates[-(test_size + 1):], extended_actual_prices[-(test_size + 1):], color='blue', linestyle='dotted')
plt.plot(extended_dates[-(test_size + 1):], extended_predictions[-(test_size + 1):], color='red', linestyle='dotted', label='Next Day Prediction')

plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

print(f"Predicted price for the next day: {next_day_prediction[0][0]}")
