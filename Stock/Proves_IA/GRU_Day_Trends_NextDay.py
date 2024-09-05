# -*- coding: utf-8 -*-
"""
GRU_Day_Trends_NextDay
PRÀCTIQUES ÈXIT 
"""
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import GRU, Dense
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Descarregar dades històriques de l'acció
stock_symbol = 'AAPL'
start_date = '2020-01-01'
end_date = '2023-12-31'
data = yf.download(stock_symbol, start=start_date, end=end_date)

# Preprocessament de les dades
close_prices = data['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
close_prices_scaled = scaler.fit_transform(close_prices)

# Crear etiquetes per amunt (1) o avall (0)
data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
labels = data['Target'].values[:-1]

# Ajustar les dades escalades per coincidir amb la longitud de les etiquetes
close_prices_scaled = close_prices_scaled[:-1]

# Crear dades d'entrada per GRU
def create_lstm_data(data, labels, time_steps=1):
    x, y = [], []
    for i in range(len(data) - time_steps):
        x.append(data[i:(i + time_steps), 0])
        y.append(labels[i + time_steps])
    return np.array(x), np.array(y)

# Establir els passos de temps i crear dades d'entrada
time_steps = 8
x, y = create_lstm_data(close_prices_scaled, labels, time_steps)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# Dividir les dades en conjunts d'entrenament i prova
train_size = int(len(x) * 0.7)
test_size = len(x) - train_size
x_train, x_test = x[0:train_size], x[train_size:len(x)]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

# Construir el model GRU
model = Sequential()
model.add(GRU(units=50, return_sequences=True, input_shape=(time_steps, 1)))
model.add(GRU(units=50))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el model
model.fit(x_train, y_train, epochs=50, batch_size=50)

# Predir les dades de prova
predictions = model.predict(x_test)
predictions = np.where(predictions > 0.5, 1, 0)

# Avaluar el model
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("Classification Report:")
print(classification_report(y_test, predictions))
print("Accuracy Score:")
print(accuracy_score(y_test, predictions))

# Representació gràfica dels resultats
plt.figure(figsize=(14, 5))
plt.plot(data.index[train_size + time_steps: train_size + time_steps + len(y_test)], y_test, color='blue', label='Actual Direction')
plt.plot(data.index[train_size + time_steps: train_size + time_steps + len(predictions)], predictions, color='red', label='Predicted Direction')
plt.title('Stock Price Movement Prediction')
plt.xlabel('Date')
plt.ylabel('Movement (0: Down, 1: Up)')
plt.legend()
plt.show()

