################################################# IMPORTACIÓ LLIBRERIES #################################################
#########################################################################################################################

import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

################################################# FUNCIONS #################################################
############################################################################################################

# Funció per llegir i processar dades dels fitxers CSV
def read_and_preprocess_csv(file_path):
    df = pd.read_csv(file_path, parse_dates=['Fecha'])
    columns_of_interest = ["Fecha", "Último", "Apertura", "Máximo", "Mínimo", "Vol.", "% var."]
    
    # Convertir les columnes numèriques a format numèric, manejant errors
    for col in columns_of_interest:
        if col != "Fecha" and col != "Vol.":
            df[col] = pd.to_numeric(df[col].str.replace('.', '').str.replace(',', '.'), errors='coerce')
            mean_value = df[col].mean(skipna=True)
            df[col] = df[col].fillna(mean_value)
    return df

# Funció per guardar el dataframe (del fitxer CSV) en un fitxer Excel
def save_to_excel(df, file_path):
    try:
        df.to_excel(file_path, index=False, engine='openpyxl')
        print(f"Dades guardades a {file_path}")
    except Exception as e:
        print(f"Error guardant el fitxer: {e}")

# Funció per crear dades d'entrada per LSTM
def create_lstm_data(stock_data, trends_data, labels, time_steps=10):
    x, y = [], []
    for i in range(len(stock_data) - time_steps):
        x.append(np.hstack((stock_data[i:(i + time_steps)], trends_data[i:(i + time_steps)])))
        y.append(labels[i + time_steps])
    return np.array(x), np.array(y)

############################################ OBTENCIÓ DE DADES ############################################
###########################################################################################################

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
stock_symbol = 'BTC'
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

# Re-samplar dades de la borsa a freqüència setmanal
data = data.resample('W').last()

# Llegir les dades de Google Trends des del fitxer CSV
trends_apple = pd.read_csv('trends_apple.csv', parse_dates=['Dia'], index_col='Dia')
trends_bitcoin = pd.read_csv('trends_bitcoin.csv', parse_dates=['Dia'], index_col='Dia')
trends_criptomoneda = pd.read_csv('trends_criptomoneda.csv', parse_dates=['Dia'], index_col='Dia')



# Fusionar dades de la borsa amb dades de tendències
data = data.merge(trends_data[[keyword]], left_index=True, right_index=True, how='left')

for key, df in data_frames.items():
    df.set_index('Fecha', inplace=True)
    data = data.merge(df[['Último']], left_index=True, right_index=True, how='left', suffixes=('', f'_{key}'))

############################################ PRE-PROCESSAT DE DADES ############################################
################################################################################################################

# Preprocessament de dades
close_prices = data['Close'].values.reshape(-1, 1)  # preu de tancament --> variable que volem predir (y)
google_trends = data[keyword].values.reshape(-1, 1)
or_prices = data['Último_or'].values.reshape(-1, 1)
gas_prices = data['Último_gasNatural'].values.reshape(-1, 1)
petroliBrent_prices = data['Último_petroliBrent'].values.reshape(-1, 1)
petroliCru_prices = data['Último_petroliCru'].values.reshape(-1, 1)
plata_prices = data['Último_plata'].values.reshape(-1, 1)

scalers = [MinMaxScaler(feature_range=(0, 1)) for _ in range(7)]
scaled_data = [scaler.fit_transform(data) for scaler, data in zip(scalers, [close_prices, google_trends, or_prices, gas_prices, petroliBrent_prices, petroliCru_prices, plata_prices])]

close_prices_scaled, google_trends_scaled, or_prices_scaled, gas_prices_scaled, petroliBrent_prices_scaled, petroliCru_prices_scaled, plata_prices_scaled = scaled_data

# Crear etiquetes per amunt (1) o avall (0)
data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)  # el desplaçament permet comparar els preus de tancament del dia actual amb els preus de tancament del dia següent
labels = data['Target'].values[:-1]

# Eliminem l'últim valor per alinear-lo amb les etiquetes.
scaled_data = [d[:-1] for d in scaled_data]

############################################ CREEM L'LSTM ############################################
######################################################################################################

# Crear les dades d'entrada per al model LSTM
time_steps = 10
x, y = create_lstm_data(*scaled_data, labels=labels, time_steps=time_steps)
x = np.reshape(x, (x.shape[0], x.shape[1], len(scaled_data)))

# Dividir les dades en conjunts d'entrenament i prova
train_size = int(len(x) * 0.8)
test_size = len(x) - train_size
x_train, x_test = x[0:train_size], x[train_size:len(x)]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

# Distribució de classes equilibrada en el conjunt d'entrenament
class_weight = {0: 1.0, 1: sum(y_train == 0) / sum(y_train == 1)}

# Construir el model LSTM
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(time_steps, len(scaled_data))))
model.add(LSTM(units=50))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el model
model.fit(x_train, y_train, epochs=100, batch_size=32, class_weight=class_weight)

# Predir amb les dades de prova
predictions = model.predict(x_test)
predictions = np.where(predictions > 0.5, 1, 0)

# Avaluar el model
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

