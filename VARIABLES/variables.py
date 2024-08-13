import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Funció per llegir i processar dades dels fitxers CSV
def read_and_preprocess_csv(file_path):
    df = pd.read_csv(file_path, parse_dates=['Fecha'])
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
    df['Último'] = clean_column(df['Último'])
    save_to_excel(df, f'C:/Users/Mar/Documents/GitHub/Practiques_UDG/VARIABLES/{key}_filtrat.xlsx')
    data_frames[key] = df

# Descarregar dades històriques de la borsa
stock_symbol = 'BTC-USD'
start_date = '2024-04-16'
end_date = '2024-07-16'

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
trends_apple = pd.read_csv('trends_apple.csv', parse_dates=['Dia'], index_col='Dia')
trends_bitcoin = pd.read_csv('trends_bitcoin.csv', parse_dates=['Dia'], index_col='Dia')
trends_criptomoneda = pd.read_csv('trends_criptomoneda.csv', parse_dates=['Dia'], index_col='Dia')

# Fusionar dades de la borsa amb dades de tendències
data = data.merge(trends_apple, left_index=True, right_index=True, how='left', suffixes=('', '_apple'))
data = data.merge(trends_bitcoin, left_index=True, right_index=True, how='left', suffixes=('', '_bitcoin'))
data = data.merge(trends_criptomoneda, left_index=True, right_index=True, how='left', suffixes=('', '_criptomoneda'))

# Fusionar dades dels preus de mercats amb el dataframe principal
for key, df in data_frames.items():
    df.set_index('Fecha', inplace=True)  # Assegura't que l'índex és la data
    df.rename(columns={'Último': f'Último_{key}'}, inplace=True)  # Renombra la columna 'Último'
    data = data.merge(df[[f'Último_{key}']], left_index=True, right_index=True, how='left')

# Eliminem els valors buits (borsa tancada --> dades que no ens serveixen per a res)
data = data.dropna()

# Preprocessament de dades
close_prices = data['Close'].values.reshape(-1, 1)  # preu de tancament --> variable que volem predir (y)
apple_trends = data['Apple'].values.reshape(-1, 1)
bitcoin_trends = data['Bitcoin'].values.reshape(-1, 1)
criptomoneda_trends = data['Criptomoneda'].values.reshape(-1, 1)
or_prices = data['Último_or'].values.reshape(-1, 1)
gas_prices = data['Último_gasNatural'].values.reshape(-1, 1)
petroliBrent_prices = data['Último_petroliBrent'].values.reshape(-1, 1)
petroliCru_prices = data['Último_petroliCru'].values.reshape(-1, 1)
plata_prices = data['Último_plata'].values.reshape(-1, 1)

# Escalar les dades
scalers = [MinMaxScaler(feature_range=(0, 1)) for _ in range(9)]
scaled_data = [scaler.fit_transform(d) for scaler, d in zip(scalers, [close_prices, apple_trends, bitcoin_trends, criptomoneda_trends, or_prices, gas_prices, petroliBrent_prices, petroliCru_prices, plata_prices])]

# Crear diccionari de les dades escalades
taula_data = {
    'close_prices_scaled': scaled_data[0].flatten(),
    'apple_trends_scaled': scaled_data[1].flatten(), 
    'bitcoin_trends_scaled': scaled_data[2].flatten(), 
    'criptomoneda_trends_scaled': scaled_data[3].flatten(), 
    'or_prices_scaled': scaled_data[4].flatten(), 
    'gas_prices_scaled': scaled_data[5].flatten(), 
    'petroliBrent_prices_scaled': scaled_data[6].flatten(), 
    'petroliCru_prices_scaled': scaled_data[7].flatten(), 
    'plata_prices_scaled': scaled_data[8].flatten()
}

# Convertir el diccionari en un DataFrame
scaled_data_df = pd.DataFrame(taula_data)

print(scaled_data_df)

# Crear etiquetes per amunt (1) o avall (0)
data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)  # el desplaçament permet comparar els preus de tancament del dia actual amb els preus de tancament del dia següent
labels = data['Target'].values[:-1]

# Eliminem l'últim valor per alinear-lo amb les etiquetes
scaled_data = [d[:-1] for d in scaled_data]

# Combinar dades d'entrada
combined_data = np.hstack(scaled_data)

