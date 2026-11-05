import random
import math
import matplotlib.pyplot as plt

# Step 1: Define cities
cities = {
    0: (0, 0),
    1: (2, 3),
    2: (5, 2),
    3: (6, 6),
    4: (8, 3)
}

n = len(cities)

# Calculate distance between cities
def distance(i, j):
    x1, y1 = cities[i]
    x2, y2 = cities[j]
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

# Step 2: Initialize parameters
num_ants = 10
iterations = 100
alpha = 1.0
beta = 2.0
rho = 0.5
Q = 100

# Initialize pheromone trails
pheromone = [[1.0] * n for _ in range(n)]

best_route = None
best_distance = float("inf")

# Step 3: Construct routes
for iteration in range(iterations):
    all_routes = []

    for ant in range(num_ants):
        start = random.randrange(n)
        route = [start]
        unvisited = set(range(n)) - {start}

        while unvisited:
            current = route[-1]
            choices = list(unvisited)

            # Calculate probabilities
            weights = []
            for city in choices:
                tau = pheromone[current][city] ** alpha
                eta = (1 / distance(current, city)) ** beta
                weights.append(tau * eta)

            next_city = random.choices(
                choices, weights=weights, k=1
            )[0]

            route.append(next_city)
            unvisited.remove(next_city)

        # Return to the starting city
        route.append(route[0])

        # Calculate route length
        length = sum(
            distance(route[i], route[i + 1])
            for i in range(n)
        )

        all_routes.append((route, length))

        if length < best_distance:
            best_distance = length
            best_route = route.copy()

    # Step 4: Evaporate pheromones
    for i in range(n):
        for j in range(n):
            pheromone[i][j] *= (1 - rho)

    # Deposit pheromones
    for route, length in all_routes:
        deposit = Q / length

        for i in range(n):
            a = route[i]
            b = route[i + 1]
            pheromone[a][b] += deposit
            pheromone[b][a] += deposit

# Step 5: Print the best route
print("Best Route:", best_route)
print("Shortest Distance:", round(best_distance, 2))

# Step 6: Draw the graph
x = [cities[i][0] for i in best_route]
y = [cities[i][1] for i in best_route]

plt.figure(figsize=(8, 6))

# Draw best route
plt.plot(x, y, 'bo-', linewidth=2, markersize=8)

# Label cities
for city, (cx, cy) in cities.items():
    plt.annotate(
        f"City {city}",
        (cx, cy),
        xytext=(5, 7),
        textcoords="offset points"
    )

plt.title(
    f"ACO for TSP\nShortest Distance: {best_distance:.2f}"
)
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.grid(True)
plt.tight_layout()
plt.show()

#Vedankitha M[1BM25CS539]