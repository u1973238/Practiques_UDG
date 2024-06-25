# Importar las bibliotecas necesarias
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Cargar los datos desde el archivo CSV
df = pd.read_csv('BRK-A.csv')

# Preprocesamiento de datos
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df[['Close']]

# Crear variables predictoras (features) y la variable objetivo (target)
# En este caso, usaremos el valor de cierre histórico como características para predecir el siguiente valor de cierre
n = 10  # Número de observaciones pasadas a considerar como features

for i in range(1, n+1):
    df[f'Close_Lag_{i}'] = df['Close'].shift(i)

# Eliminar filas con NaN resultantes de la creación de lagged features
df.dropna(inplace=True)

# Definir X (features) y y (target)
X = df.drop(columns=['Close'])
y = df['Close']

# Dividir los datos en conjuntos de entrenamiento y prueba (70% entrenamiento, 30% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inicializar el modelo de Regresión con Bosques Aleatorios (Random Forest)
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular el error cuadrático medio (MSE) en el conjunto de prueba
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Para hacer una predicción futura
# Supongamos que queremos predecir el siguiente día después de los datos existentes
# Primero, necesitamos las últimas 'n' observaciones para formar las características de predicción
last_n_obs = df.tail(n)[[f'Close_Lag_{i}' for i in range(1, n+1)]].values.reshape(1, -1)

# Realizar la predicción del siguiente día
next_day_prediction = model.predict(last_n_obs)
print(f'Predicción para el siguiente día: {next_day_prediction[0]}')
