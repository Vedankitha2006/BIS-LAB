import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score
)

# ==========================================
# 1. LOAD DATASET
# ==========================================

try:
    data = pd.read_csv("fetal_health.csv")
except FileNotFoundError:
    print("ERROR: fetal_health.csv not found!")
    print("Save the dataset in the same folder as GWO.py")
    raise SystemExit

data.columns = data.columns.str.strip().str.lower()

if "fetal_health" not in data.columns:
    print("ERROR: Dataset must contain fetal_health column.")
    raise SystemExit

data = data.dropna()

X = data.drop("fetal_health", axis=1).values
y = data["fetal_health"].astype(int).values

print("Dataset loaded successfully!")
print("Total records:", len(data))
print("Total features:", X.shape[1])

# ==========================================
# 2. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Validation data for GWO
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train, y_train,
    test_size=0.25,
    random_state=42,
    stratify=y_train
)

# ==========================================
# 3. FITNESS FUNCTION
# ==========================================

def fitness(position):

    # Convert wolf position into selected features
    selected = position > 0.5

    if np.sum(selected) == 0:
        return 1.0

    model = LogisticRegression(max_iter=2000)

    model.fit(X_fit[:, selected], y_fit)

    predictions = model.predict(X_val[:, selected])

    score = f1_score(
        y_val,
        predictions,
        average="macro"
    )

    penalty = 0.01 * (
        np.sum(selected) / X.shape[1]
    )

    return 1 - score + penalty


# ==========================================
# 4. INITIALIZE GWO
# ==========================================

np.random.seed(42)

num_wolves = 15
iterations = 30
dimensions = X_train.shape[1]

wolves = np.random.uniform(
    0, 1,
    (num_wolves, dimensions)
)

best_history = []

global_best = None
global_best_fitness = float("inf")

# ==========================================
# 5. GREY WOLF OPTIMIZATION
# ==========================================

for iteration in range(iterations):

    fitness_values = np.array([
        fitness(wolf) for wolf in wolves
    ])

    # Identify Alpha, Beta and Delta wolves
    indices = np.argsort(fitness_values)

    alpha = wolves[indices[0]].copy()
    beta = wolves[indices[1]].copy()
    delta = wolves[indices[2]].copy()

    # Store global best solution
    if fitness_values[indices[0]] < global_best_fitness:
        global_best_fitness = fitness_values[indices[0]]
        global_best = alpha.copy()

    best_history.append(global_best_fitness)

    # Decrease control parameter
    a = 2 - (2 * iteration / iterations)

    # Update wolf positions
    for i in range(num_wolves):

        positions = []

        for leader in [alpha, beta, delta]:

            r1 = np.random.rand(dimensions)
            r2 = np.random.rand(dimensions)

            A = 2 * a * r1 - a
            C = 2 * r2

            D = np.abs(C * leader - wolves[i])

            new_position = leader - A * D

            positions.append(new_position)

        wolves[i] = np.mean(positions, axis=0)

        # Keep positions between 0 and 1
        wolves[i] = np.clip(wolves[i], 0, 1)

    print(
        "Iteration:", iteration + 1,
        "| Best Fitness:", round(global_best_fitness, 4)
    )

# ==========================================
# 6. SELECT BEST FEATURES
# ==========================================

selected_features = global_best > 0.5

# Ensure at least one feature is selected
if not np.any(selected_features):
    selected_features[np.argmax(global_best)] = True

feature_names = data.drop(
    "fetal_health", axis=1
).columns

print("\nSelected Features:")

for name in feature_names[selected_features]:
    print("-", name)

print(
    "\nNumber of selected features:",
    np.sum(selected_features)
)

# ==========================================
# 7. TRAIN FINAL MODEL
# ==========================================

model = LogisticRegression(max_iter=2000)

model.fit(
    X_train[:, selected_features],
    y_train
)

predictions = model.predict(
    X_test[:, selected_features]
)

accuracy = accuracy_score(y_test, predictions)

print("\n========== GWO FETAL HEALTH RESULTS ==========")

print("Best Fitness:", round(global_best_fitness, 4))

print("Test Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")

print(classification_report(
    y_test,
    predictions,
    labels=[1, 2, 3],
    target_names=["Normal", "Suspect", "Pathological"],
    zero_division=0
))

# ==========================================
# 8. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    predictions,
    labels=[1, 2, 3]
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Normal",
        "Suspect",
        "Pathological"
    ]
)

display.plot(cmap="Blues")

plt.title("Fetal Health Classification using GWO")
plt.tight_layout()
plt.show()

# ==========================================
# 9. GWO CONVERGENCE GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, iterations + 1),
    best_history,
    marker="o"
)

plt.title("GWO Optimization Progress")
plt.xlabel("Iterations")
plt.ylabel("Best Fitness")
plt.grid(True)
plt.tight_layout()
plt.show()

# ==========================================
# 10. ACCURACY GRAPH
# ==========================================

plt.figure(figsize=(7, 5))

plt.bar(
    ["GWO Feature Selection"],
    [accuracy * 100]
)

plt.ylabel("Accuracy (%)")
plt.title("Fetal Health Prediction Accuracy")
plt.ylim(0, 100)
plt.tight_layout()
plt.show()

print("\nPROGRAM COMPLETED SUCCESSFULLY!")
