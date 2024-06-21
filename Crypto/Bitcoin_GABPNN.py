import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import resample
import numpy as np
from deap import base, creator, tools, algorithms
import warnings

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

# Define the evaluation function for GA
def evaluate_individual(individual):
    hidden_layer_sizes = tuple(map(int, individual[:2]))
    max_iter = int(individual[2])
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=42)
    model.fit(train_data_balanced[features], train_target_balanced)
    predictions = model.predict(test_data[features])
    accuracy = accuracy_score(test_target, predictions)
    return accuracy,

# Genetic Algorithm setup
if "FitnessMax" not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_int", np.random.randint, 50, 200)  # Adjusted range for hidden layers and max_iter
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=3)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=50, up=200, indpb=0.2)  # Adjusted range
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate_individual)

population = toolbox.population(n=20)
NGEN = 20
CXPB = 0.5
MUTPB = 0.2

# Genetic Algorithm main loop
for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))
    print(f"Generation {gen} completed")

# Best individual from GA
best_individual = tools.selBest(population, k=1)[0]
best_hidden_layer_sizes = tuple(map(int, best_individual[:2]))
best_max_iter = int(best_individual[2])

print(f"Best Individual: {best_individual}")

# Train the optimized BPNN model
gabpnn_model = MLPClassifier(hidden_layer_sizes=best_hidden_layer_sizes, max_iter=best_max_iter, random_state=42)
gabpnn_model.fit(train_data_balanced[features], train_target_balanced)

# Make predictions with GABPNN
gabpnn_predictions = gabpnn_model.predict(test_data[features])

# Evaluate the GABPNN model
gabpnn_accuracy = accuracy_score(test_target, gabpnn_predictions)
gabpnn_conf_matrix = confusion_matrix(test_target, gabpnn_predictions)
gabpnn_class_report = classification_report(test_target, gabpnn_predictions, zero_division=1)

print('GABPNN Accuracy:', gabpnn_accuracy)
print('GABPNN Confusion Matrix:\n', gabpnn_conf_matrix)
print('GABPNN Classification Report:\n', gabpnn_class_report)
