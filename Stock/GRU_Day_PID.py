import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import yfinance as yf

# Load stock data
stock_symbol = 'AAPL'
start_date = '2020-01-01'
end_date = '2023-12-31'
data = yf.download(stock_symbol, start=start_date, end=end_date)

# Load PID data
pid_data = pd.read_csv('pib_filtered.csv')

# Melt the PID dataframe to have a Date column
pid_data = pd.melt(pid_data, id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
                   var_name='Year', value_name='PID')

# Filter data
country_code = 'USA'
filtered_pid_data = pid_data[(pid_data['Country Code'] == country_code)]

# Ensure stock data has a 'Date' column and set it as datetime
data.reset_index(inplace=True)
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year

# Map PID values to stock data
pid_dict = filtered_pid_data.set_index('Year')['PID'].to_dict()
data['PID'] = data['Year'].map(pid_dict)
data.drop(columns=['Year'], inplace=True)

# Extract and scale relevant columns
close_prices = data['Close'].values.reshape(-1, 1)
pid_values = data['PID'].values.reshape(-1, 1)

scaler_price = MinMaxScaler(feature_range=(0, 1))
close_prices_scaled = scaler_price.fit_transform(close_prices)

scaler_pid = MinMaxScaler(feature_range=(0, 1))
pid_scaled = scaler_pid.fit_transform(pid_values)

# Create labels for price movement
data['Next_Close'] = data['Close'].shift(-1)
data['Movement'] = np.where(data['Next_Close'] > data['Close'], 1, 0)
labels = data['Movement'].values[:-1]

# Adjust data lengths
close_prices_scaled = close_prices_scaled[:-1]
pid_scaled = pid_scaled[:-1]

# Create input data
def create_gru_data(prices_data, pid_data, labels, time_steps=1):
    x, y = [], []
    for i in range(len(prices_data) - time_steps):
        combined_data = np.hstack((prices_data[i:(i + time_steps)], pid_data[i:(i + time_steps)]))
        x.append(combined_data)
        y.append(labels[i + time_steps])
    return np.array(x), np.array(y)

# Settings
time_steps = 10
x, y = create_gru_data(close_prices_scaled, pid_scaled, labels, time_steps)
x = np.reshape(x, (x.shape[0], x.shape[1], 2))

# Split data
train_size = int(len(x) * 0.9)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the GRU Model
model = Sequential()
model.add(GRU(units=50, return_sequences=True, input_shape=(time_steps, 2)))
model.add(Dropout(0.2))  # Dropout layer
model.add(GRU(units=50))
model.add(Dropout(0.2))  # Dropout layer
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.1, callbacks=[early_stopping])

# Predict on the test data
predictions = model.predict(x_test)
predictions = np.where(predictions > 0.5, 1, 0)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("Classification Report:")
print(classification_report(y_test, predictions))
print("Accuracy Score:")
print(accuracy_score(y_test, predictions))
print("ROC AUC Score:")
print(roc_auc_score(y_test, predictions))

# Plotting the training history
plt.figure(figsize=(14, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotting the results
plt.figure(figsize=(14, 5))
plt.plot(data.index[train_size + time_steps: train_size + time_steps + len(y_test)], y_test, color='blue', label='Actual Direction')
plt.plot(data.index[train_size + time_steps: train_size + time_steps + len(predictions)], predictions, color='red', label='Predicted Direction')
plt.title('Stock Price Movement Prediction')
plt.xlabel('Date')
plt.ylabel('Movement (0: Down, 1: Up)')
plt.legend()
plt.show()

