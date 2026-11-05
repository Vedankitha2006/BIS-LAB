import random
import numpy as np
import matplotlib.pyplot as plt

# Sample training data: house size and price
X = np.array([500, 700, 900, 1100, 1300], dtype=float)
y = np.array([100, 140, 180, 220, 260], dtype=float)

# Objective function: Mean Squared Error
def fitness(solution):
    slope, intercept = solution
    predictions = slope * X + intercept
    return np.mean((y - predictions) ** 2)

# Cuckoo Search parameters
num_nests = 15
Pa = 0.25
iterations = 100

# Initialize nests: [slope, intercept]
nests = [
    [random.uniform(0, 1), random.uniform(-100, 100)]
    for _ in range(num_nests)
]

# Find the best nest
best_nest = min(nests, key=fitness).copy()
best_errors = []

# Cuckoo Search
for iteration in range(iterations):

    # Generate new solutions using random steps
    for i in range(num_nests):
        step = np.random.normal(0, 0.1, 2)
        new_nest = np.array(nests[i]) + step

        # Keep parameters within limits
        new_nest[0] = np.clip(new_nest[0], 0, 1)
        new_nest[1] = np.clip(new_nest[1], -100, 100)

        j = random.randrange(num_nests)

        if fitness(new_nest) < fitness(nests[j]):
            nests[j] = new_nest.tolist()

    # Abandon some nests and create new solutions
    for i in range(num_nests):
        if random.random() < Pa:
            nests[i] = [
                random.uniform(0, 1),
                random.uniform(-100, 100)
            ]

    # Update the best solution
    current_best = min(nests, key=fitness)

    if fitness(current_best) < fitness(best_nest):
        best_nest = current_best.copy()

    best_errors.append(fitness(best_nest))

# Display results
slope, intercept = best_nest

print("Best Slope:", round(slope, 4))
print("Best Intercept:", round(intercept, 4))
print("Minimum MSE:", round(fitness(best_nest), 4))

# Plot actual and predicted prices
plt.scatter(X, y, label="Actual Prices")
plt.plot(X, slope * X + intercept,
         color="red", label="Predicted Prices")
plt.xlabel("House Size")
plt.ylabel("House Price")
plt.title("Cuckoo Search for House Price Prediction")
plt.legend()
plt.grid(True)
plt.show()

# Plot optimization progress
plt.plot(best_errors)
plt.xlabel("Iteration")
plt.ylabel("Best Mean Squared Error")
plt.title("Cuckoo Search Optimization Progress")
plt.grid(True)
plt.show()

#Vedankitha M[1BM25CS539]