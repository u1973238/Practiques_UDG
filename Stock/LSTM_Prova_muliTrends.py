import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Download Historical Stock Data
stock_symbol = 'AAPL'
start_date = '2023-07-01'
end_date = '2024-07-01'
data = yf.download(stock_symbol, start=start_date, end=end_date)

# Data Preprocessing
data['Pct_Change'] = data['Close'].shift(-1) - data['Close']
data = data.dropna()

# Extract the labels (price differences)
labels = data['Pct_Change'].shift(-1).dropna().values

# Normalize the labels (price differences) to the range [-1, 1]
scaler_change = MinMaxScaler(feature_range=(-1, 1))
labels_scaled = scaler_change.fit_transform(labels.reshape(-1, 1))

# Prepare the closing prices for input data
close_prices = data['Close'].values.reshape(-1, 1)
scaler_prices = MinMaxScaler(feature_range=(0, 1))
close_prices_scaled = scaler_prices.fit_transform(close_prices)

# Generate synthetic Google Trends data for demonstration
# Replace this with actual Google Trends data
np.random.seed(42)  # This is just to ensure the synthetic data is reproducible, can be removed for actual data
'FALTA QUE AGAFA EL TRENS, ARA MATEIX ES RANDOM'
google_trends_data = {
    'Apple': np.random.rand(),
    'AAPL': np.random.rand(len(data)),
    'iPhone': np.random.rand(len(data)),
    'MacBook': np.random.rand(len(data)),
    'Tim Cook': np.random.rand(len(data)),
    'Samsung': np.random.rand(len(data)),
    'Apple store': np.random.rand(len(data)),
    'Apple Watch': np.random.rand(len(data)),
    'iPad': np.random.rand(len(data)),
    'Apple services': np.random.rand(len(data)),
    'Apple support': np.random.rand(len(data)),
    'Apple reviews': np.random.rand(len(data)),
    'Apple accessories': np.random.rand(len(data)),
}
google_trends_df = pd.DataFrame(google_trends_data, index=data.index)
'''
# Normalize Google Trends data
scaler_trends = MinMaxScaler(feature_range=(0, 1))
google_trends_scaled = scaler_trends.fit_transform(google_trends_df)

# Combine the stock prices and Google Trends data
combined_data = np.hstack((close_prices_scaled[:-1], google_trends_scaled[:-1]))  # Exclude last day for labels

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

'''