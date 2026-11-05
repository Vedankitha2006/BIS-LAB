import random

# Fitness function
def fitness(x):
    weight = x * 10

    if x < 5:
        return 0

    return 1 / weight

# Parameters
population_size = 10
generations = 50
mutation_rate = 0.1

# Initial population
population = [random.randint(1, 20)
              for _ in range(population_size)]

# Genetic Algorithm
for g in range(generations):

    # Selection
    population.sort(key=fitness, reverse=True)
    parents = population[:5]

    # Crossover
    new_population = parents.copy()

    while len(new_population) < population_size:

        p1, p2 = random.sample(parents, 2)

        child = (p1 + p2) // 2

        # Mutation
        if random.random() < mutation_rate:
            child = random.randint(1, 20)

        new_population.append(child)

    population = new_population

# Best solution
best = max(population, key=fitness)

print("Optimal Beam Size:", best)
print("Minimum Weight:", best * 10)

#Vedankitha M[1BM25CS539]