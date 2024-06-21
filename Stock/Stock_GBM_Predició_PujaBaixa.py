import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Cargar los datos desde el archivo CSV
df = pd.read_csv('BRK-A.csv')

# Preprocesamiento de datos
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df[['Close']]

# Calcular el cambio porcentual diario en el precio de cierre
df['Daily_Return'] = df['Close'].pct_change()

# Etiquetar si el precio subió o bajó (1 para subir, 0 para bajar)
df['Price_Up'] = np.where(df['Daily_Return'] > 0, 1, 0)

# Eliminar filas con NaN resultantes de los cambios porcentuales
df.dropna(inplace=True)

# Mostrar las primeras filas del dataframe
print(df.head())

# Definir X (features) y y (target)
X = df[['Close']].values  # Usar solo el precio de cierre como característica
y = df['Price_Up'].values  # La variable target es si el precio subió o bajó

# Dividir los datos en entrenamiento y prueba (usando las últimas 30% instancias para prueba)
train_size = 0.7
split_index = int(train_size * len(X))

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Inicializar el modelo GBM para clasificación binaria
params = {
    'objective': 'binary:logistic',   # Objetivo de clasificación binaria
    'learning_rate': 0.1,             # Tasa de aprendizaje
    'max_depth': 5,                   # Profundidad máxima del árbol
    'subsample': 0.8,                 # Submuestra de la observación
    'colsample_bytree': 0.8,          # Submuestra de columnas por árbol
    'n_estimators': 100               # Número de árboles a construir
}
model = xgb.XGBClassifier(**params)

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Mostrar el reporte de clasificación
print(classification_report(y_test, y_pred))

# Obtener las últimas 30% instancias de datos para predicción
X_predict = X[-int(0.3 * len(X)):]

# Realizar predicciones
predictions = model.predict(X_predict)

# Calcular y mostrar el porcentaje de veces que se predice que el precio subirá
percent_rise = np.mean(predictions)
print(f'Porcentaje de predicciones que el precio subirá: {percent_rise * 100:.2f}%')
