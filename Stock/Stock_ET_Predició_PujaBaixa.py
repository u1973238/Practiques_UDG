import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Cargar los datos desde el archivo CSV
df = pd.read_csv('AAPL.csv')

# Preprocesamiento de datos
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Calcular el cambio porcentual diario en el precio de cierre
df['Daily_Return'] = df['Close'].pct_change()

# Etiquetar si el precio subió o bajó (1 para subir, 0 para bajar)
df['Price_Up'] = np.where(df['Daily_Return'] > 0, 1, 0)

# Eliminar filas con NaN resultantes de los cambios porcentuales
df.dropna(inplace=True)

# Definir X (features) y y (target)
# Usar todas las columnas excepto 'Price_Up' y 'Daily_Return' como características
X = df.drop(columns=['Price_Up', 'Daily_Return']).values
y = df['Price_Up'].values

# Dividir los datos en entrenamiento y prueba (70% entrenamiento, 30% prueba)
train_size = 0.95
split_index = int(train_size * len(X))

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Inicializar el modelo Extra Trees Classifier
model = ExtraTreesClassifier(n_estimators=100, random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

