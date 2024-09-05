# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:41:18 2024

@author: Mar
"""

import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

# Funció per llegir i processar dades dels fitxers CSV
def read_and_preprocess_csv(file_path):
    df = pd.read_csv(file_path, parse_dates=['Fecha'], dayfirst=True)  # Afegit dayfirst=True
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
    # Aquí pots carregar un fitxer local com a alternativa
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
    df.set_index('Fecha', inplace=True)  # Assegura't que l'índex és la data
    df.index = pd.to_datetime(df.index)  # Convertir a DatetimeIndex si no ho és
    df.rename(columns={'Último': f'Último_{key}'}, inplace=True)  # Renombra la columna 'Último'
    data = data.merge(df[[f'Último_{key}']], left_index=True, right_index=True, how='left')

# Eliminem els valors buits (borsa tancada --> dades que no ens serveixen per a res)
data = data.dropna()

# Re-samplar les dades a freqüència setmanal
data_weekly = data.resample('W').last()  # Utilitzem l'últim valor de la setmana (preu de tancament del divendres)

# Preprocessament de dades (les dades ja estan processades, només les resamplem a setmanes)
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

# Llista de variables escalades que sí existeixen
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

# Convertim la llista a un array numpy per poder utilitzar-la
data_scaled = np.hstack(data_scaled)

# Crear un dataframe amb totes les dades escalades per a la matriu de correlació
data_scaled_df = pd.DataFrame(data_scaled, columns=variables.keys())

# Calcular la matriu de correlació
corr_mat = data_scaled_df.corr()

# Crear etiquetes binàries de moviments de preus (+1 si el preu setmanal va pujar, 0 si va baixar)
price_diff = np.diff(close_prices_weekly, axis=0)  # Canvi en el preu setmanal
labels = (price_diff > 0).astype(int)  # 1 si puja, 0 si baixa

# Dividir les dades en entrenament i prova
X, y = create_lstm_data(data_scaled[:-1], labels)  # Excloem l'últim per a y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Modificar l'arquitectura del model
model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))  # Afegeixo més dropout per evitar sobreajustament
model.add(LSTM(units=32, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=16))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))

# Compilar el model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el model
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.2, verbose=1)

# Avaluar el model
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)  # Convertir probabilitats a etiquetes

# Mètriques de rendiment
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

# Visualitzar l'històric de la pèrdua i l'exactitud amb plotly
fig_loss = go.Figure()
fig_loss.add_trace(go.Scatter(y=history.history['loss'], mode='lines', name='Train Loss'))
fig_loss.add_trace(go.Scatter(y=history.history['val_loss'], mode='lines', name='Validation Loss'))
fig_loss.update_layout(title='Loss', xaxis_title='Epochs', yaxis_title='Loss')

fig_accuracy = go.Figure()
fig_accuracy.add_trace(go.Scatter(y=history.history['accuracy'], mode='lines', name='Train Accuracy'))
fig_accuracy.add_trace(go.Scatter(y=history.history['val_accuracy'], mode='lines', name='Validation Accuracy'))
fig_accuracy.update_layout(title='Accuracy', xaxis_title='Epochs', yaxis_title='Accuracy')

fig_loss.show()
fig_accuracy.show()
