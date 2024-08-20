# -*- coding: utf-8 -*-

import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
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

# Reemplaçar comas i punts en les columnes
def clean_column(column):
    column = column.str.replace('.', '', regex=False)  # Eliminar punts que separen milers
    column = column.str.replace(',', '.', regex=False)  # Substituir comes que separen decimals per punts
    return pd.to_numeric(column, errors='coerce')  # Convertir a float

# Rutes als fitxers CSV
files = {
    'or': 'preu_or_1any.csv',
    'gasNatural': 'preu_gasNatural_1any.csv',
    'petroliCru': 'preu_petroliCru_1any.csv',
    'plata': 'preu_plata_1any.csv'
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
start_date = '2022-08-12'
end_date = '2024-08-12'

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
trends_apple = pd.read_csv('trends_apple_1any.csv', parse_dates=['Setmana'], index_col='Setmana')
trends_bitcoin = pd.read_csv('trends_bitcoin_1any.csv', parse_dates=['Setmana'], index_col='Setmana')
trends_criptomoneda = pd.read_csv('trends_criptomoneda_1any.csv', parse_dates=['Setmana'], index_col='Setmana')

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

# Escalar les dades setmanals
scalers = [MinMaxScaler() for _ in range(8)]  # Crear un escalador per a cada conjunt de dades
scaled_data_weekly = [scaler.fit_transform(d) for scaler, d in zip(scalers, [close_prices_weekly, apple_trends_weekly, bitcoin_trends_weekly, criptomoneda_trends_weekly, or_prices_weekly, gas_prices_weekly, petroliCru_prices_weekly, plata_prices_weekly])]

# Crear diccionari de les dades escalades setmanals
taula_data_weekly = {
    'close_prices_scaled': scaled_data_weekly[0].flatten(),
    'apple_trends_scaled': scaled_data_weekly[1].flatten(), 
    'bitcoin_trends_scaled': scaled_data_weekly[2].flatten(), 
    'criptomoneda_trends_scaled': scaled_data_weekly[3].flatten(), 
    'or_prices_scaled': scaled_data_weekly[4].flatten(), 
    'gas_prices_scaled': scaled_data_weekly[5].flatten(), 
    'petroliCru_prices_scaled': scaled_data_weekly[6].flatten(), 
    'plata_prices_scaled': scaled_data_weekly[7].flatten()
}

# Convertir el diccionari en un DataFrame
scaled_data_df_weekly = pd.DataFrame(taula_data_weekly)

# Calcular la matriu de correlació (opcional)
corr_mat = scaled_data_df_weekly.corr()

# Crear etiquetes per amunt (1) o avall (0) amb dades setmanals
data_weekly['Target'] = np.where(data_weekly['Close'].shift(-1) > data_weekly['Close'], 1, 0)
labels_weekly = data_weekly['Target'].values[:-1]

# Eliminem l'últim valor per alinear-lo amb les etiquetes
scaled_data_weekly = [d[:-1] for d in scaled_data_weekly]

# Combinar dades d'entrada setmanals
combined_data_weekly = np.hstack(scaled_data_weekly)

# No cal ajustar la finestra temporal amb SVM; utilitzarem directament les dades setmanals combinades
x_weekly = combined_data_weekly
y_weekly = labels_weekly

# Dividir les dades en entrenament i prova
x_train_weekly, x_test_weekly, y_train_weekly, y_test_weekly = train_test_split(
    x_weekly, y_weekly, test_size=0.2, random_state=42
)

# Crear el model SVM
model_svm = SVC(kernel='rbf', C=1.0, gamma='scale')

# Entrenar el model amb les dades d'entrenament
model_svm.fit(x_train_weekly, y_train_weekly)

# Predir amb les dades de prova
predictions_weekly = model_svm.predict(x_test_weekly)

# Avaluar el model
print("Confusion Matrix:")
print(confusion_matrix(y_test_weekly, predictions_weekly))
print("Classification Report:")
print(classification_report(y_test_weekly, predictions_weekly))
print("Accuracy Score:")
print(accuracy_score(y_test_weekly, predictions_weekly))

# Mostrar gràfic de les prediccions versus les etiquetes reals
plt.figure(figsize=(10, 5))
plt.plot(y_test_weekly, label='Reals')
plt.plot(predictions_weekly, label='Prediccions', linestyle='dashed')
plt.title('Reals vs Prediccions')
plt.xlabel('Índex de prova')
plt.ylabel('Classe')
plt.legend()
plt.show()
