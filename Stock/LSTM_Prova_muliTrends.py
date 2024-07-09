import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Read the CSV file
combined_data = pd.read_csv('merged_data_with_normalized_target.csv',index_col=0)

# Normalize the input features and the target separately
feature_columns = combined_data.columns.difference(['target'])
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

# Fit the scaler on the features and transform
combined_data[feature_columns] = scaler_features.fit_transform(combined_data[feature_columns])

# Creating Input Data for LSTM
def create_lstm_data(data, labels, time_steps=1):
    x, y = [], []
    for i in range(len(data) - time_steps):
        x.append(data[i:(i + time_steps)])
        y.append(labels[i + time_steps - 1])  # Align labels with time steps
    return np.array(x), np.array(y)

# Setting Time Steps and Creating Input Data
time_steps = 10
labels_scaled = scaler_target.fit_transform(combined_data['target'].values.reshape(-1, 1))

# Convert dataframe to numpy array
data = combined_data[feature_columns].values

# Create LSTM data
x, y = create_lstm_data(data, labels_scaled, time_steps)

# Split data into training and testing sets
train_size = int(len(x) * 0.8)
x_train, y_train = x[:train_size], y[:train_size]
x_test, y_test = x[train_size:], y[train_size:]

# Building the LSTM Model for Regression
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_steps, x.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the Model
model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Predicting on the Test Data
predictions_scaled = model.predict(x_test)

# Inverse transform the predictions to get back to original scale
predictions = scaler_target.inverse_transform(predictions_scaled)
y_test_inverse = scaler_target.inverse_transform(y_test)

# Plotting the results
plt.figure(figsize=(14, 5))
plt.plot(range(train_size + time_steps, train_size + time_steps + len(y_test)), y_test_inverse, color='blue', label='Actual Change')
plt.plot(range(train_size + time_steps, train_size + time_steps + len(predictions)), predictions, color='red', label='Predicted Change')
plt.title('Next Day Stock Price Change Prediction with Google Trends')
plt.xlabel('Time Step')
plt.ylabel('Price Change')
plt.legend()
plt.show()
