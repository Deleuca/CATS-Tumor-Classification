import sys
import random
import numpy as np
from deap import base, creator, tools, algorithms

sys.path.insert(0, "src")
from logistic_regression import read_data, logistic_regression

# GA hyperparameters
POPULATION_SIZE = 50
N_GENERATIONS   = 40
CROSSOVER_PROB  = 0.7
MUTATION_PROB   = 0.2
TOURNAMENT_SIZE = 3
FLIP_PROB       = 0.05   # per-gene mutation probability


def evaluate(individual, X, y):
    selected = [i for i, bit in enumerate(individual) if bit == 1]
    if not selected:
        return (0.0,)
    X_subset = X.iloc[:, selected]
    cv_scores = logistic_regression(X_subset, y)
    return (cv_scores.mean(),)


def build_toolbox(n_features, X, y):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bit",    random.randint, 0, 1)
    toolbox.register("individual",  tools.initRepeat, creator.Individual,
                     toolbox.attr_bit, n=n_features)
    toolbox.register("population",  tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate",    evaluate, X=X, y=y)
    toolbox.register("mate",        tools.cxTwoPoint)
    toolbox.register("mutate",      tools.mutFlipBit, indpb=FLIP_PROB)
    toolbox.register("select",      tools.selTournament, tournsize=TOURNAMENT_SIZE)
    return toolbox


def run_ga(X, y):
    toolbox    = build_toolbox(n_features=X.shape[1], X=X, y=y)
    population = toolbox.population(n=POPULATION_SIZE)

    stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats.register("mean", np.mean)
    stats.register("max",  np.max)

    hall_of_fame = tools.HallOfFame(maxsize=1)

    population, log = algorithms.eaSimple(
        population,
        toolbox,
        cxpb=CROSSOVER_PROB,
        mutpb=MUTATION_PROB,
        ngen=N_GENERATIONS,
        stats=stats,
        halloffame=hall_of_fame,
        verbose=True,
    )

    best_individual = hall_of_fame[0]
    selected_features = [i for i, bit in enumerate(best_individual) if bit == 1]
    best_accuracy = best_individual.fitness.values[0]

    return selected_features, best_accuracy


if __name__ == "__main__":
    X, y = read_data()
    selected_features, best_accuracy = run_ga(X, y)
    print(f"\nSelected {len(selected_features)} features out of {X.shape[1]}")
    print(f"Best CV accuracy: {best_accuracy:.4f}")
