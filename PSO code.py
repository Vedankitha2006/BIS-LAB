import random
import math

# Start, target and obstacle
start = (0, 0)
target = (10, 10)
obstacle = (5, 5)
radius = 2

# Fitness function
def fitness(x):
    distance = math.dist(start, (x[0], x[1]))
    distance += math.dist((x[0], x[1]), target)

    obstacle_dist = math.dist((x[0], x[1]), obstacle)

    if obstacle_dist < radius:
        distance += 100

    return distance

# PSO parameters
n = 20
iterations = 100
w = 0.5
c1 = 1.5
c2 = 1.5

# Initialize particle positions and velocities
position = [[random.uniform(0, 10),
             random.uniform(0, 10)] for _ in range(n)]

velocity = [[random.uniform(-1, 1),
              random.uniform(-1, 1)] for _ in range(n)]

pbest = [p[:] for p in position]
pbest_value = [fitness(p) for p in position]

best_index = pbest_value.index(min(pbest_value))
gbest = pbest[best_index][:]

# PSO iterations
for _ in range(iterations):
    for i in range(n):
        for d in range(2):
            r1 = random.random()
            r2 = random.random()

            velocity[i][d] = (
                w * velocity[i][d]
                + c1 * r1 * (pbest[i][d] - position[i][d])
                + c2 * r2 * (gbest[d] - position[i][d])
            )

            position[i][d] += velocity[i][d]
            position[i][d] = max(0, min(10, position[i][d]))

        value = fitness(position[i])

        if value < pbest_value[i]:
            pbest[i] = position[i][:]
            pbest_value[i] = value

        if value < fitness(gbest):
            gbest = position[i][:]

# Display results
print("Best Waypoint:", [round(v, 2) for v in gbest])
print("Path Fitness:", round(fitness(gbest), 2))

#Vedankitha M[1BM25CS539]