import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from pytrends.request import TrendReq
import time

# Function to fetch Google Trends data with retry mechanism
def fetch_google_trends_data(keyword, start_date, end_date):
    pytrends = TrendReq(hl='en-US', tz=360)
    pytrends.build_payload([keyword], timeframe=f'{start_date} {end_date}')
    trends_data = None
    while trends_data is None or trends_data.empty:
        try:
            trends_data = pytrends.interest_over_time()
            if trends_data.empty:
                print("Google Trends data is empty, retrying in 60 seconds...")
                time.sleep(60)
        except Exception as e:
            print(f"Error: {e}, retrying in 60 seconds...")
            time.sleep(60)
    return trends_data[keyword]

# Downloading Historical Stock Data
stock_symbol = 'AAPL'
start_date = '2020-01-01'
end_date = '2024-01-01'
data = yf.download(stock_symbol, start=start_date, end=end_date)

# Fetching Google Trends data
keyword = 'Apple'
trends_data = fetch_google_trends_data(keyword, start_date, end_date)

# Check and log if trends data is empty
if trends_data.empty:
    raise ValueError(f"Google Trends data for keyword '{keyword}' is empty.")

# Forward-fill the Google Trends data to daily frequency
trends_data = trends_data.reindex(pd.date_range(start=start_date, end=end_date)).fillna(method='ffill')

# Merging stock data with trends data
data = data.merge(trends_data, left_index=True, right_index=True)

# Data Preprocessing
close_prices = data['Close'].values.reshape(-1, 1)
google_trends = data[keyword].values.reshape(-1, 1)

# Check if google_trends data is empty after merging
if len(google_trends) == 0:
    raise ValueError("Google Trends data is empty after merging.")

scaler_close = MinMaxScaler(feature_range=(0, 1))
scaler_trends = MinMaxScaler(feature_range=(0, 1))
close_prices_scaled = scaler_close.fit_transform(close_prices)
google_trends_scaled = scaler_trends.fit_transform(google_trends)

# Creating labels for up (1) or down (0)
data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
labels = data['Target'].values[:-1]

# Adjust scaled data to match the length of labels
close_prices_scaled = close_prices_scaled[:-1]
google_trends_scaled = google_trends_scaled[:-1]

# Creating Input Data for LSTM
def create_lstm_data(stock_data, trends_data, labels, time_steps=1):
    x, y = [], []
    for i in range(len(stock_data) - time_steps):
        x.append(np.hstack((stock_data[i:(i + time_steps)], trends_data[i:(i + time_steps)])))
        y.append(labels[i + time_steps])
    return np.array(x), np.array(y)

# Setting Time Steps and Creating Input Data
time_steps = 10
x, y = create_lstm_data(close_prices_scaled, google_trends_scaled, labels, time_steps)
x = np.reshape(x, (x.shape[0], x.shape[1], 2))

# Splitting the data into training and test sets
train_size = int(len(x) * 0.9)
test_size = len(x) - train_size
x_train, x_test = x[0:train_size], x[train_size:len(x)]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

# Building the LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_steps, 2)))
model.add(LSTM(units=50))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the Model
model.fit(x_train, y_train, epochs=50, batch_size=32)

# Predicting on the Test Data
predictions = model.predict(x_test)
predictions = np.where(predictions > 0.5, 1, 0)

# Evaluating the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("Classification Report:")
print(classification_report(y_test, predictions))
print("Accuracy Score:")
print(accuracy_score(y_test, predictions))

# Plotting the results
plt.figure(figsize=(14, 5))
plt.plot(data.index[train_size + time_steps: train_size + time_steps + len(y_test)], y_test, color='blue', label='Actual Direction')
plt.plot(data.index[train_size + time_steps: train_size + time_steps + len(predictions)], predictions, color='red', label='Predicted Direction')
plt.title('Stock Price Movement Prediction with Google Trends')
plt.xlabel('Date')
plt.ylabel('Movement (0: Down, 1: Up)')
plt.legend()
plt.show()
