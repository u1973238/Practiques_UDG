import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from pytrends.request import TrendReq
import time

# Funció per llegir i processar dades dels fitxers CSV
def read_and_preprocess_csv(file_path):
    df = pd.read_csv(file_path, parse_dates=['Fecha'])
    columns_of_interest = ["Fecha", "Último", "Apertura", "Máximo", "Mínimo", "Vol.", "% var."]
    
    # Convertir les columnes numèriques a format numèric, manejant errors
    for col in columns_of_interest:
        if col != "Fecha" and col != "Vol.":
            df[col] = pd.to_numeric(df[col].str.replace('.', '').str.replace(',', '.'), errors='coerce')
            mean_value = df[col].mean(skipna=True)
            df[col] = df[col].fillna(mean_value)  # Canvi per evitar FutureWarning
    return df

# Funció per guardar el dataframe en un fitxer Excel
def save_to_excel(df, file_path):
    try:
        df.to_excel(file_path, index=False, engine='openpyxl')
        print(f"Dades guardades a {file_path}")
    except Exception as e:
        print(f"Error guardant el fitxer: {e}")

# Funció per obtenir dades de Google Trends
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
    # Aquí pots carregar un fitxer local com a alternativa
    data = pd.read_csv('path_to_local_file.csv', parse_dates=['Date'], index_col='Date')

# Assegurar-se que l'índex és de tipus DatetimeIndex
if not isinstance(data.index, pd.DatetimeIndex):
    data.index = pd.to_datetime(data.index)

# Re-samplar dades de la borsa a freqüència setmanal
data = data.resample('W').last()

# Obtenir dades de Google Trends
keyword = 'Apple'
trends_data = fetch_google_trends_data(keyword, start_date, end_date)

# Comprovar i registrar si les dades de tendències són buides
if trends_data.empty:
    raise ValueError(f"Google Trends data for keyword '{keyword}' is empty.")

# Fusionar dades de la borsa amb dades de tendències
data = data.merge(trends_data, left_index=True, right_index=True)

# Preprocessament de dades
close_prices = data['Close'].values.reshape(-1, 1)
google_trends = data[keyword].values.reshape(-1, 1)

# Comprovar si les dades de tendències de Google estan buides després de fusionar
if len(google_trends) == 0:
    raise ValueError("Google Trends data is empty after merging.")

scaler_close = MinMaxScaler(feature_range=(0, 1))
scaler_trends = MinMaxScaler(feature_range=(0, 1))
close_prices_scaled = scaler_close.fit_transform(close_prices)
google_trends_scaled = scaler_trends.fit_transform(google_trends)

# Crear etiquetes per amunt (1) o avall (0)
data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
labels = data['Target'].values[:-1]

# Ajustar les dades escalades per coincidir amb la longitud de les etiquetes
close_prices_scaled = close_prices_scaled[:-1]
google_trends_scaled = google_trends_scaled[:-1]

# Funció per crear dades d'entrada per LSTM
def create_lstm_data(stock_data, trends_data, labels, time_steps=10):
    x, y = [], []
    for i in range(len(stock_data) - time_steps):
        x.append(np.hstack((stock_data[i:(i + time_steps)], trends_data[i:(i + time_steps)])))
        y.append(labels[i + time_steps])
    return np.array(x), np.array(y)

# Configuració dels passos de temps i creació de les dades d'entrada
time_steps = 10
x, y = create_lstm_data(close_prices_scaled, google_trends_scaled, labels, time_steps)
x = np.reshape(x, (x.shape[0], x.shape[1], 2))

# Dividir les dades en conjunts d'entrenament i prova
train_size = int(len(x) * 0.8)
test_size = len(x) - train_size
x_train, x_test = x[0:train_size], x[train_size:len(x)]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

# Distribució de classes equilibrada en el conjunt d'entrenament
class_weight = {0: 1.0, 1: sum(y_train == 0) / sum(y_train == 1)}

# Construir el model LSTM
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(time_steps, 2)))
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
print("Classification Report:")
print(classification_report(y_test, predictions))
print("Accuracy Score:")
print(accuracy_score(y_test, predictions))

# Representació gràfica dels resultats
plt.figure(figsize=(14, 5))
plt.plot(data.index[train_size + time_steps: train_size + time_steps + len(y_test)], y_test, color='blue', label='Actual Direction')
plt.plot(data.index[train_size + time_steps: train_size + time_steps + len(predictions)], predictions, color='red', label='Predicted Direction')
plt.title('Stock Price Movement Prediction with Google Trends')
plt.xlabel('Date')
plt.ylabel('Movement (0: Down, 1: Up)')
plt.legend()
plt.show()

