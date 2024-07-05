import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import resample

# Load data
data = pd.read_csv("BTC-USD.csv", delimiter=",")

# Convert the Date column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Ensure data is sorted by date
data.sort_values('Date', inplace=True)

# Compute EMA (Exponential Moving Average)
data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()

# Drop rows with NaN values
data.dropna(inplace=True)

# Select features and target
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'EMA_12']
target = (data['Close'].shift(-1) > data['Close']).astype(int)  # 1 if the next day's 'Close' price is higher, else 0
data = data.dropna()  # Drop rows with NaN values created by the shift() function

# Split data
train_data = data.iloc[:int(0.70 * len(data)), :]
test_data = data.iloc[int(0.70 * len(data)):, :]
train_target = target.iloc[:int(0.70 * len(target))]
test_target = target.iloc[int(0.70 * len(target)):]

# Check class distribution in the training target
print("Class distribution in training target:")
print(train_target.value_counts())

# Handle class imbalance by upsampling the minority class
train_data_0 = train_data[train_target == 0]
train_data_1 = train_data[train_target == 1]
train_target_0 = train_target[train_target == 0]
train_target_1 = train_target[train_target == 1]

# Upsample minority class
train_data_1_upsampled = resample(train_data_1,
                                  replace=True,  # sample with replacement
                                  n_samples=len(train_data_0),  # to match majority class
                                  random_state=42)  # reproducible results

train_target_1_upsampled = resample(train_target_1,
                                    replace=True,  # sample with replacement
                                    n_samples=len(train_target_0),  # to match majority class
                                    random_state=42)  # reproducible results

# Combine majority class with upsampled minority class
train_data_balanced = pd.concat([train_data_0, train_data_1_upsampled])
train_target_balanced = pd.concat([train_target_0, train_target_1_upsampled])

print("Balanced class distribution in training target:")
print(train_target_balanced.value_counts())

# Define and train the BPNN model
bpnn_model = MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=500, random_state=42)
bpnn_model.fit(train_data_balanced[features], train_target_balanced)

# Make predictions with BPNN
bpnn_predictions = bpnn_model.predict(test_data[features])

# Evaluate the BPNN model
bpnn_accuracy = accuracy_score(test_target, bpnn_predictions)
bpnn_conf_matrix = confusion_matrix(test_target, bpnn_predictions)
bpnn_class_report = classification_report(test_target, bpnn_predictions, zero_division=1)

print('BPNN Accuracy:', bpnn_accuracy)
print('BPNN Confusion Matrix:\n', bpnn_conf_matrix)
print('BPNN Classification Report:\n', bpnn_class_report)
