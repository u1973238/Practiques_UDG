import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score

# Load data
data = pd.read_csv("BTC-USD.csv", delimiter=",")

# Convert the Date column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Ensure data is sorted by date
data.sort_values('Date', inplace=True)

# Compute EMA 26 (Exponential Moving Average)
data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()

# Drop rows with NaN values
data.dropna(inplace=True)

# Select features and target
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'EMA_26']
target = (data['Close'].shift(-1) > data['Close']).astype(int)  # 1 if the next day's 'Close' price is higher, else 0
data = data.dropna()  # Drop rows with NaN values created by the shift() function

# Split data
train_data = data.iloc[:int(0.70 * len(data)), :]
test_data = data.iloc[int(0.70 * len(data)):, :]
train_target = target.iloc[:int(0.70 * len(target))]
test_target = target.iloc[int(0.70 * len(target)):]

# Define the evaluation function for GA
def evaluate_individual(individual):
    hidden_layer_sizes = tuple(map(int, individual[:2]))
    max_iter = int(individual[2])
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=42)
    model.fit(train_data[features], train_target)
    predictions = model.predict(test_data[features])
    accuracy = accuracy_score(test_target, predictions)
    return accuracy,

# Genetic Algorithm setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_int", np.random.randint, 10, 100)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=3)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=10, up=100, indpb=0.2)
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

# Train the optimized GANN model
gann_model = MLPClassifier(hidden_layer_sizes=best_hidden_layer_sizes, max_iter=best_max_iter, random_state=42)
gann_model.fit(train_data[features], train_target)

# Make predictions with GANN
gann_predictions = gann_model.predict(test_data[features])

# Evaluate the GANN model
gann_accuracy = accuracy_score(test_target, gann_predictions)
gann_conf_matrix = confusion_matrix(test_target, gann_predictions)
gann_class_report = classification_report(test_target, gann_predictions, zero_division=0)

# Calculate precision, recall, and F1-score
precision = precision_score(test_target, gann_predictions, zero_division=0)
recall = recall_score(test_target, gann_predictions, zero_division=0)
f1 = f1_score(test_target, gann_predictions, zero_division=0)

print('GANN Accuracy:', gann_accuracy)
print('GANN Confusion Matrix:\n', gann_conf_matrix)
print('GANN Classification Report:\n', gann_class_report)
print('GANN Precision:', precision)
print('GANN Recall:', recall)
print('GANN F1 Score:', f1)
