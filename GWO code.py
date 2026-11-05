
import numpy as np
import matplotlib.pyplot as plt

# Healthcare optimization using GWO
# Objective: Minimize prediction error

def fitness(x):
    return (x - 5) ** 2

# Parameters
wolves = 10
iterations = 50

# Initialize wolf positions
positions = np.random.uniform(-10, 10, wolves)

alpha = 0
beta = 0
delta = 0

alpha_score = float("inf")
beta_score = float("inf")
delta_score = float("inf")

convergence = []

# GWO algorithm
for t in range(iterations):

    for i in range(wolves):
        score = fitness(positions[i])

        if score < alpha_score:
            delta_score = beta_score
            delta = beta

            beta_score = alpha_score
            beta = alpha

            alpha_score = score
            alpha = positions[i]

        elif score < beta_score:
            delta_score = beta_score
            delta = beta

            beta_score = score
            beta = positions[i]

        elif score < delta_score:
            delta_score = score
            delta = positions[i]

    a = 2 - t * (2 / iterations)

    for i in range(wolves):
        r1, r2 = np.random.rand(), np.random.rand()
        A1 = 2 * a * r1 - a
        C1 = 2 * r2
        D_alpha = abs(C1 * alpha - positions[i])
        X1 = alpha - A1 * D_alpha

        r1, r2 = np.random.rand(), np.random.rand()
        A2 = 2 * a * r1 - a
        C2 = 2 * r2
        D_beta = abs(C2 * beta - positions[i])
        X2 = beta - A2 * D_beta

        r1, r2 = np.random.rand(), np.random.rand()
        A3 = 2 * a * r1 - a
        C3 = 2 * r2
        D_delta = abs(C3 * delta - positions[i])
        X3 = delta - A3 * D_delta

        positions[i] = (X1 + X2 + X3) / 3

    convergence.append(alpha_score)

# Results
print("Healthcare Optimization using GWO")
print("----------------------------------")
print("Best solution:", round(alpha, 4))
print("Minimum prediction error:", round(alpha_score, 6))

# Graph
plt.plot(convergence)
plt.xlabel("Iterations")
plt.ylabel("Prediction Error")
plt.title("GWO Healthcare Optimization")
plt.grid()
plt.show()
