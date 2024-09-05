# -*- coding: utf-8 -*-
"""
GRU_Day_Trrends_NextDay.py

@author: Mar
"""

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import GRU, Dense, BatchNormalization, Dropout
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# Downloading Historical Stock Data
stock_symbol = 'AAPL'
start_date = '2020-01-01'
end_date = '2023-12-31'
data = yf.download(stock_symbol, start=start_date, end=end_date)

# Data Preprocessing
close_prices = data['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
close_prices_scaled = scaler.fit_transform(close_prices)

# Creating labels for up (1) or down (0)
data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
labels = data['Target'].values[:-1]

# Adjust scaled data to match the length of labels
close_prices_scaled = close_prices_scaled[:-1]

# Creating Input Data for GRU
def create_gru_data(data, labels, time_steps=1):
    x, y = [], []
    for i in range(len(data) - time_steps):
        x.append(data[i:(i + time_steps), 0])
        y.append(labels[i + time_steps])
    return np.array(x), np.array(y)

# Setting Time Steps and Creating Input Data
time_steps = 10
x, y = create_gru_data(close_prices_scaled, labels, time_steps)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# Splitting the data into training and test sets
train_size = int(len(x) * 0.9)
test_size = len(x) - train_size
x_train, x_test = x[0:train_size], x[train_size:len(x)]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

model = Sequential()
# Primera capa GRU
model.add(GRU(units=100, return_sequences=True, input_shape=(time_steps, 1)))
model.add(BatchNormalization())
model.add(Dropout(0.2))  # Capa de Dropout per evitar l'overfitting
# Segona capa GRU
model.add(GRU(units=100, return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.2))  # Capa de Dropout per evitar l'overfitting
# Tercera capa GRU
model.add(GRU(units=100))
model.add(BatchNormalization())
model.add(Dropout(0.2))  # Capa de Dropout per evitar l'overfitting
# Capes denses finals
model.add(Dense(units=50, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
# Configurar l'optimitzador amb un learning rate ajustat
learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)

# Compile the model with the custom optimizer
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Training the Model
#history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.1)
# Training the Model with more epochs
history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.1)

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

# Plotting the training history
plt.figure(figsize=(14, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotting the results
plt.figure(figsize=(14, 5))
plt.plot(data.index[train_size + time_steps: train_size + time_steps + len(y_test)], y_test, color='blue', label='Actual Direction')
plt.plot(data.index[train_size + time_steps: train_size + time_steps + len(predictions)], predictions, color='red', label='Predicted Direction')
plt.title('Stock Price Movement Prediction')
plt.xlabel('Date')
plt.ylabel('Movement (0: Down, 1: Up)')
plt.legend()
plt.show()
