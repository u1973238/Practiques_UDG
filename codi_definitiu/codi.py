# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 11:13:59 2024

@author: Mar
"""

import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf
import matplotlib.pyplot as plt

# Funció per llegir i processar dades dels fitxers CSV
def read_and_preprocess_csv(file_path):
    df = pd.read_csv(file_path, parse_dates=['Fecha'], dayfirst=True)
    return df

# Funció per guardar el dataframe (del fitxer CSV) en un fitxer Excel
def save_to_excel(df, file_path):
    try:
        df.to_excel(file_path, index=False, engine='openpyxl')
        print(f"Dades guardades a {file_path}")
    except Exception as e:
        print(f"Error guardant el fitxer: {e}")

# Funció per crear dades d'entrada per LSTM
def create_lstm_data(stock_data, labels, time_steps=10):
    x, y = [], []
    for i in range(len(stock_data) - time_steps):
        x.append(stock_data[i:(i + time_steps)])
        y.append(labels[i + time_steps])
    return np.array(x), np.array(y)

# Reemplaçar comas i punts en les columnes
def clean_column(column):
    column = column.str.replace('.', '', regex=False)  # Eliminar punts que separen milers
    column = column.str.replace(',', '.', regex=False)  # Substituir comes que separen decimals per punts
    return pd.to_numeric(column, errors='coerce')  # Convertir a float

# Rutes als fitxers CSV
files = {
    'or': 'preu_or_5anys.csv',
    'gasNatural': 'preu_gasNatural_5anys.csv',
    'petroliCru': 'preu_petroliCru_5anys.csv',
    'plata': 'preu_plata_5anys.csv',
    'plati': 'preu_plati_5anys.csv',
    'coure': 'preu_coure_5anys.csv'
}

# Processar cada fitxer
data_frames = {}
for key, file in files.items():
    df = read_and_preprocess_csv(file)
    df['Último'] = clean_column(df['Último'])
    save_to_excel(df, f'C:/Users/Mar/Documents/GitHub/Practiques_UDG/VARIABLES/{key}_filtrat.xlsx')
    data_frames[key] = df

# Descarregar dades històriques de la borsa
stock_symbol = 'BTC-USD'
start_date = '2019-08-22'
end_date = '2024-08-22'

try:
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    if data.empty:
        raise ValueError(f"No data found for symbol {stock_symbol}.")
except Exception as e:
    print(f"Error downloading data: {e}")
    data = pd.read_csv('path_to_local_file.csv', parse_dates=['Date'], index_col='Date')

# Assegurar-se que l'índex és de tipus DatetimeIndex
if not isinstance(data.index, pd.DatetimeIndex):
    data.index = pd.to_datetime(data.index)

# Re-samplar dades de la borsa a freqüència diària
data = data.resample('D').ffill()

# Llegir les dades de Google Trends des del fitxer CSV
trends_apple = pd.read_csv('trends_apple_5anys.csv', parse_dates=['Setmana'], index_col='Setmana')
trends_bitcoin = pd.read_csv('trends_bitcoin_5anys.csv', parse_dates=['Setmana'], index_col='Setmana')
trends_criptomoneda = pd.read_csv('trends_criptomoneda_5anys.csv', parse_dates=['Setmana'], index_col='Setmana')

# Assegurar-se que l'índex de Google Trends és un DatetimeIndex abans de fusionar
trends_apple.index = pd.to_datetime(trends_apple.index)
trends_bitcoin.index = pd.to_datetime(trends_bitcoin.index)
trends_criptomoneda.index = pd.to_datetime(trends_criptomoneda.index)

# Fusionar dades de la borsa amb dades de tendències
data = data.merge(trends_apple, left_index=True, right_index=True, how='left', suffixes=('', '_apple'))
data = data.merge(trends_bitcoin, left_index=True, right_index=True, how='left', suffixes=('', '_bitcoin'))
data = data.merge(trends_criptomoneda, left_index=True, right_index=True, how='left', suffixes=('', '_criptomoneda'))

# Fusionar dades dels preus de mercats amb el dataframe principal
for key, df in data_frames.items():
    df.set_index('Fecha', inplace=True)
    df.index = pd.to_datetime(df.index)
    df.rename(columns={'Último': f'Último_{key}'}, inplace=True)
    data = data.merge(df[[f'Último_{key}']], left_index=True, right_index=True, how='left')

# Eliminem els valors buits (borsa tancada --> dades que no ens serveixen per a res)
data = data.dropna()

# Re-samplar les dades a freqüència setmanal
data_weekly = data.resample('W').last()

# Preprocessament de dades
close_prices_weekly = data_weekly['Close'].values.reshape(-1, 1)
apple_trends_weekly = data_weekly['apple: (Arreu del món)'].values.reshape(-1, 1)
bitcoin_trends_weekly = data_weekly['bitcoin: (Arreu del món)'].values.reshape(-1, 1)
criptomoneda_trends_weekly = data_weekly['criptomoneda: (Arreu del món)'].values.reshape(-1, 1)
or_prices_weekly = data_weekly['Último_or'].values.reshape(-1, 1)
gas_prices_weekly = data_weekly['Último_gasNatural'].values.reshape(-1, 1)
petroliCru_prices_weekly = data_weekly['Último_petroliCru'].values.reshape(-1, 1)
plata_prices_weekly = data_weekly['Último_plata'].values.reshape(-1, 1)
plati_prices_weekly = data_weekly['Último_plati'].values.reshape(-1, 1)
coure_prices_weekly = data_weekly['Último_coure'].values.reshape(-1, 1)

# Escalar les dades setmanals entre 0 i 1
scalers = {}
data_scaled = []

variables = {
    'Close': close_prices_weekly,
    'apple_trends': apple_trends_weekly,
    'bitcoin_trends': bitcoin_trends_weekly,
    'criptomoneda_trends': criptomoneda_trends_weekly,
    'or': or_prices_weekly,
    'gasNatural': gas_prices_weekly,
    'petroliCru': petroliCru_prices_weekly,
    'plata': plata_prices_weekly,
    'plati': plati_prices_weekly,
    'coure': coure_prices_weekly
}

for key, value in variables.items():
    scaler = MinMaxScaler()
    data_scaled.append(scaler.fit_transform(value))
    scalers[key] = scaler

data_scaled = np.hstack(data_scaled)
data_scaled_df = pd.DataFrame(data_scaled, columns=variables.keys())

# Calcular la matriu de correlació
corr_mat = data_scaled_df.corr()

# Crear etiquetes binàries de moviments de preus
price_diff = np.diff(close_prices_weekly, axis=0)
labels = (price_diff > 0).astype(int)

# Dividir les dades en entrenament i prova
X, y = create_lstm_data(data_scaled[:-1], labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Funció per crear el model LSTM amb opcions addicionals
def create_model(units=128, dropout_rate=0.2, optimizer='adam', learning_rate=0.001, num_layers=4):
    model = Sequential()
    
    for i in range(num_layers):
        return_sequences = (i < num_layers - 1)
        model.add(LSTM(units=units // (2**i), return_sequences=return_sequences, input_shape=(X_train.shape[1], X_train.shape[2]) if i == 0 else None))
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1, activation='sigmoid'))
    
    if optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == 'adagrad':
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    elif optimizer == 'nadam':
        optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
    
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Configurar el KerasClassifier per a GridSearchCV
model = KerasClassifier(build_fn=create_model, epochs=50, batch_size=64, verbose=1)

# Definir la graella de paràmetres per a GridSearchCV
param_grid = {
    'units': [8, 16, 32, 64, 128, 256],
    'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
    'optimizer': ['adam', 'rmsprop', 'sgd', 'adagrad', 'nadam'],
    'batch_size': [32, 64, 128],
    'epochs': [20, 30, 50, 60],
    'learning_rate': [0.001, 0.01, 0.1],
    'num_layers': [2, 3, 4]
}

# Configurar GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1)

# Ajustar el model
grid_result = grid.fit(X_train, y_train)

# Mostrar els millors paràmetres i puntuació
print(f"Best Score: {grid_result.best_score_}")
print(f"Best Parameters: {grid_result.best_params_}")

# Avaluar el millor model
best_model = grid_result.best_estimator_
y_pred = best_model.predict(X_test)

# Mostrar els resultats de la predicció
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy Score: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")

# Funció per guardar els resultats
def save_results(filename, best_score, best_params, accuracy):
    with open(filename, 'w') as f:
        f.write(f"Best score from Grid Search: {best_score}\n")
        f.write(f"Best parameters: {best_params}\n")
        f.write(f"Accuracy Score: {accuracy}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(f"{conf_matrix}\n")
        f.write("\nClassification Report:\n")
        f.write(f"{class_report}\n")

# Guardar els resultats en un fitxer
save_results('resultats_model.txt', grid_result.best_score_, grid_result.best_params_, accuracy)

