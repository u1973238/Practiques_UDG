import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Cargar los datos desde el archivo CSV
df = pd.read_csv('AAPL.csv')

# Preprocesamiento de datos
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Definir X (features) y y (target)
# Usar todas las columnas excepto 'Close' como características
X = df.drop(columns=['Close']).values
y = df['Close'].values

# Dividir los datos en entrenamiento y prueba (70% entrenamiento, 30% prueba)
train_size = 0.95
split_index = int(train_size * len(X))

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Inicializar el modelo Extra Trees Regressor
model = ExtraTreesRegressor(n_estimators=100, random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular el error cuadrático medio (MSE) y el coeficiente de determinación (R^2)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')
print(f'R^2 Score: {r2:.2f}')

# Graficar las predicciones y los valores reales
plt.figure(figsize=(14, 7))
plt.plot(df.index[-len(y_test):], y_test, label='Valor Real', marker='o', linestyle='-')
plt.plot(df.index[-len(y_test):], y_pred, label='Predicción', marker='x', linestyle='--')
plt.title('Predicciones vs Valor Real')
plt.xlabel('Fecha')
plt.ylabel('Precio de Cierre')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

