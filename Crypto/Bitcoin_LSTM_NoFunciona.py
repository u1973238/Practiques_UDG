import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam

# Carregar les dades de BTC-USD.csv
data = pd.read_csv("BTC-USD.csv", parse_dates=['Date'])
data.sort_values('Date', inplace=True)
data.set_index('Date', inplace=True)

# Afegir l'EMA 26
data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()

# Seleccionar les característiques i la variable objectiu (Close)
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'EMA_26']
target = 'Close'

# Normalitzar les dades
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[features])

# Preparar les dades per LSTM
def create_dataset(dataset, target, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :]
        X.append(a)
        Y.append(target[i + look_back])
    return np.array(X), np.array(Y)

# Definir el look-back (número de períodes anteriors per a les dades d'entrada)
look_back = 5

# Crear les dades d'entrenament i test
train_size = int(len(scaled_data) * 0.70)
test_size = len(scaled_data) - train_size
train, test = scaled_data[0:train_size, :], scaled_data[train_size:len(scaled_data), :]

trainX, trainY = create_dataset(train, train[:, 3], look_back)
testX, testY = create_dataset(test, test[:, 3], look_back)

# Reformular l'entrada a [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], len(features)))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], len(features)))

# Definir el model LSTM
model = Sequential()
model.add(LSTM(units=50, input_shape=(look_back, len(features))))
model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Entrenar el model
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# Fer prediccions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Invertir la normalització per obtenir prediccions reals
trainPredict = scaler.inverse_transform(np.concatenate((trainX[:, -1, :][:, [3]], trainPredict), axis=1))[:, 1]
trainY = scaler.inverse_transform(trainY.reshape(-1, 1))[:, 0]
testPredict = scaler.inverse_transform(np.concatenate((testX[:, -1, :][:, [3]], testPredict), axis=1))[:, 1]
testY = scaler.inverse_transform(testY.reshape(-1, 1))[:, 0]

# Calcular l'error quadràtic mitjà
trainScore = np.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = np.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))

# Visualitzar resultats
plt.figure(figsize=(14, 7))
plt.plot(trainPredict, label='Train Predictions')
plt.plot(testPredict, label='Test Predictions')
plt.plot(data['Close'].values[look_back:], label='Actual Close Price')
plt.title('Bitcoin Price Prediction with LSTM')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()
