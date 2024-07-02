import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Downloading Historical Bitcoin Data
crypto_symbol = 'BTC-USD'
start_date = '2020-01-01'
end_date = '2024-01-01'
data = yf.download(crypto_symbol, start=start_date, end=end_date)

# Data Preprocessing
close_prices = data['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
close_prices_scaled = scaler.fit_transform(close_prices)

# Creating labels for up (1) or down (0)
data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
labels = data['Target'].values[:-1]

# Adjust scaled data to match the length of labels
close_prices_scaled = close_prices_scaled[:-1]

# Creating Input Data for LSTM
def create_lstm_data(data, labels, time_steps=1):
    x, y = [], []
    for i in range(len(data) - time_steps):
        x.append(data[i:(i + time_steps), 0])
        y.append(labels[i + time_steps])
    return np.array(x), np.array(y)

# Setting Time Steps and Creating Input Data
time_steps = 10
x, y = create_lstm_data(close_prices_scaled, labels, time_steps)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# Splitting the data into training and test sets
train_size = int(len(x) * 0.9)
test_size = len(x) - train_size
x_train, x_test = x[0:train_size], x[train_size:len(x)]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

# Building the LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_steps, 1)))
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
plt.title('Bitcoin Price Movement Prediction')
plt.xlabel('Date')
plt.ylabel('Movement (0: Down, 1: Up)')
plt.legend()
plt.show()
