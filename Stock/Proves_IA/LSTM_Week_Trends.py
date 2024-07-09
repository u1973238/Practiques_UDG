import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('merged_data_with_normalized_target.csv', index_col=0)

# Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))

# Select the features and the target
features = data.drop(['target', 'normalized_target', 'Close'], axis=1)
target = data['Close'].values.reshape(-1, 1)

# Scale the features
scaled_features = scaler.fit_transform(features)

# Ensure the target is not scaled
scaler_target = MinMaxScaler(feature_range=(0, 1))
scaled_target = scaler_target.fit_transform(target)

# Create sequences for the LSTM model
sequence_length = 60
X = []
y = []

for i in range(sequence_length, len(scaled_features)):
    X.append(scaled_features[i-sequence_length:i])
    y.append(scaled_target[i])

X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Make predictions
predictions = model.predict(X_test)

# Rescale the predictions back to original scale
predictions_rescaled = scaler_target.inverse_transform(predictions)

# Rescale the actual target values back to the original scale
actual_prices = scaler_target.inverse_transform(y_test)

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(actual_prices, color='blue', label='Actual Prices')
plt.plot(predictions_rescaled, color='red', label='Predicted Prices')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
