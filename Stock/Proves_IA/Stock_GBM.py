'''
Gradient Boosting Machine
'''
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Cargar los datos desde el archivo CSV
df = pd.read_csv('AAPL.csv')

# Preprocesamiento de datos
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df[['Close']]

# Crear variables para predecir
n = 10  # Número de observaciones pasadas a considerar como características

for i in range(1, n+1):
    df[f'Close_Lag_{i}'] = df['Close'].shift(i)

df.dropna(inplace=True)

# Definir X (features) y y (target)
X = df.drop(columns=['Close'])
y = df['Close']

# Dividir los datos en conjuntos de entrenamiento y prueba
train_size = 0.99
split_index = int(train_size * len(df))

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Inicializar el modelo GBM
params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.1,
    'max_depth': 5,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 100
}
model = xgb.XGBRegressor()

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular el error cuadrático medio (MSE) en el conjunto de prueba
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Para hacer una predicción futura
last_n_obs = df.tail(n)['Close'].values.reshape(1, -1)
next_day_prediction = model.predict(last_n_obs)
print(f'Predicción para el siguiente día: {next_day_prediction[0]}')

# Graficar las predicciones y los valores reales
plt.figure(figsize=(12, 6))
plt.plot(df.index[-len(y_test):], y_test, label='Valor Real', marker='o')
plt.plot(df.index[-len(y_test):], y_pred, label='Predicción', marker='x')
plt.title('Predicciones vs Valor Real (Conjunto de prueba)')
plt.xlabel('Fecha')
plt.ylabel('Precio de Cierre')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Obtener todas las predicciones y valores reales
y_pred_all = model.predict(X)
y_actual_all = y.values

# Graficar las predicciones y los valores reales sobre la línea de tiempo completa
plt.figure(figsize=(14, 7))
plt.plot(df.index, y_actual_all, label='Valor Real', marker='o', linestyle='-')
plt.plot(df.index, y_pred_all, label='Predicción', marker='x', linestyle='--')
plt.title('Predicciones vs Valor Real (Línea de tiempo completa)')
plt.xlabel('Fecha')
plt.ylabel('Precio de Cierre')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
