# -*- coding: utf-8 -*-
"""
codi sense google trends
"""


import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Funció per llegir i processar dades dels fitxers CSV
def read_and_preprocess_csv(file_path):
    df = pd.read_csv(file_path, parse_dates=['Fecha'])
    columns_of_interest = ["Fecha", "Último", "Apertura", "Máximo", "Mínimo", "Vol.", "% var."]
    
    for col in columns_of_interest:
        if col != "Fecha" and col != "Vol.":
            df[col] = pd.to_numeric(df[col].str.replace('.', '').str.replace(',', '.'), errors='coerce')
            mean_value = df[col].mean(skipna=True)
            df[col] = df[col].fillna(mean_value)
    return df

# Funció per guardar el dataframe en un fitxer Excel
def save_to_excel(df, file_path):
    try:
        df.to_excel(file_path, index=False, engine='openpyxl')
        print(f"Dades guardades a {file_path}")
    except Exception as e:
        print(f"Error guardant el fitxer: {e}")

# Rutes als fitxers CSV
files = {
    'or': 'preu_or.csv',
    'gasNatural': 'preu_gasNatural.csv',
    'petroliBrent': 'preu_petroliBrent.csv',
    'petroliCru': 'preu_petroliCru.csv',
    'plata': 'preu_plata.csv'
}

# Processar cada fitxer
data_frames = {}
for key, file in files.items():
    df = read_and_preprocess_csv(file)
    save_to_excel(df, f'C:/Users/Mar/Documents/GitHub/Practiques_UDG/VARIABLES/{key}_filtrat.xlsx')
    data_frames[key] = df

# Descarregar dades històriques de la borsa
stock_symbol = 'AAPL'
start_date = '2024-04-16'
end_date = '2024-07-16'

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

# Re-samplar dades de la borsa a freqüència setmanal
data = data.resample('W').last()

# Preprocessament de dades
close_prices = data['Close'].values.reshape(-1, 1)

scaler_close = MinMaxScaler(feature_range=(0, 1))
close_prices_scaled = scaler_close.fit_transform(close_prices)

# Crear etiquetes per amunt (1) o avall (0)
data['Target'] = np.where(data['Close'].shift(-1




