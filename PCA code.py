
import numpy as np
import matplotlib.pyplot as plt

from concurrent.futures import ThreadPoolExecutor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

# --------------------------------
# 1. Load Pattern Recognition Data
# --------------------------------

iris = load_iris()
X = iris.data
y = iris.target

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25,
    random_state=42, stratify=y_train
)

# --------------------------------
# 2. Initialize Cellular Parameters
# --------------------------------

grid_size = 5
iterations = 20
num_cells = grid_size * grid_size

rng = np.random.default_rng(42)

# Each cell represents a possible K value for KNN
cells = rng.integers(1, 16, size=(grid_size, grid_size))

best_k = 1
best_accuracy = 0
convergence = []

# --------------------------------
# 3. Fitness Evaluation
# --------------------------------

def fitness(k):
    model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=int(k))
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    return accuracy_score(y_val, predictions)


def evaluate_population(grid):
    # Evaluate all cells in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        scores = list(executor.map(fitness, grid.flatten()))

    return np.array(scores).reshape(grid.shape)


# --------------------------------
# 4. Parallel Cellular Algorithm
# --------------------------------

for iteration in range(iterations):

    # Evaluate fitness of every cell
    scores = evaluate_population(cells)

    # Find best cell
    index = np.unravel_index(np.argmax(scores), scores.shape)

    if scores[index] > best_accuracy:
        best_accuracy = scores[index]
        best_k = cells[index]

    # Store best accuracy
    convergence.append(best_accuracy)

    # Create new population
    new_cells = cells.copy()

    # Update each cell using its neighbors
    for i in range(grid_size):
        for j in range(grid_size):

            neighbors = []

            # Check eight neighboring cells
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:

                    if di == 0 and dj == 0:
                        continue

                    ni = (i + di) % grid_size
                    nj = (j + dj) % grid_size

                    neighbors.append((ni, nj))

            # Find the best neighboring cell
            best_neighbor = max(
                neighbors,
                key=lambda pos: scores[pos]
            )

            neighbor_k = cells[best_neighbor]

            # Update cell with random exploration
            if rng.random() < 0.2:
                new_k = rng.integers(1, 16)
            else:
                new_k = neighbor_k + rng.choice([-1, 0, 1])

            new_cells[i, j] = np.clip(new_k, 1, 15)

    # Update all cells together
    cells = new_cells

    print(
        "Iteration:", iteration + 1,
        "| Best K:", best_k,
        "| Accuracy:", round(best_accuracy * 100, 2), "%"
    )


# --------------------------------
# 5. Final Pattern Recognition
# --------------------------------

final_model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=int(best_k))
)

# Train using all available training data
X_final = np.vstack((X_train, X_val))
y_final = np.concatenate((y_train, y_val))

final_model.fit(X_final, y_final)

# Predict test patterns
y_pred = final_model.predict(X_test)

# --------------------------------
# 6. Display Results
# --------------------------------

test_accuracy = accuracy_score(y_test, y_pred)

print("\nPARALLEL CELLULAR ALGORITHM RESULTS")
print("-----------------------------------")
print("Best K value:", best_k)
print("Best Validation Accuracy:", round(best_accuracy * 100, 2), "%")
print("Test Accuracy:", round(test_accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))

# --------------------------------
# 7. Confusion Matrix
# --------------------------------

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=iris.target_names,
    cmap="Blues"
)

plt.title("Pattern Recognition Confusion Matrix")
plt.tight_layout()
plt.show()

# --------------------------------
# 8. Convergence Graph
# --------------------------------

plt.plot(
    range(1, iterations + 1),
    convergence,
    marker="o"
)

plt.xlabel("Iterations")
plt.ylabel("Best Validation Accuracy")
plt.title("Parallel Cellular Algorithm Convergence")
plt.grid()
plt.tight_layout()
plt.show()

print("\nProgram executed successfully!")
