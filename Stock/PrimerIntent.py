import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Load data
data = pd.read_csv("NVDA.csv", delimiter=",")

# Split data
train_data = data.iloc[:int(0.70 * len(data)), :]
test_data = data.iloc[int(0.70 * len(data)):, :]

# Define features and target
features = ['Open', 'Volume', 'High', 'Low', 'Close', 'Adj Close']
target = 'Close'

# Initialize and train the model
model = xgb.XGBRegressor()
model.fit(train_data[features], train_data[target])

# Make predictions
predictions = model.predict(test_data[features])

# Evaluate the model
mse = mean_squared_error(test_data[target], predictions)
Accuracy = model.score(test_data[features],test_data[target])

print('Mean Squared Error:', mse)
print('Accuracy:', Accuracy)

# Plotting predictions vs actual values
plt.plot(data['Close'],label = 'Close Price')
plt.plot(test_data[target].index, predictions, label='Predictions')
plt.legend()
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(test_data.index, test_data[target], label='Actual', color='blue')
plt.plot(test_data.index, predictions, label='Predicted', color='red')
plt.xlabel('Index')
plt.ylabel('Close Price')
plt.title('Actual vs Predicted Close Prices')
plt.legend()
plt.show()