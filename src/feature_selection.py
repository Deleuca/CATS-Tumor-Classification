"""GA-based feature selection.

The GA represents a feature subset as a length-p bit string. Fitness is the
mean CV accuracy of a classifier trained on the selected probes. The CV
harness in src/evaluate.py is shared across all callers, so any model wrapped
here is compared on the *exact same folds* as the baseline runs.

Use `run_ga(make_model, X, y, splits, ...)` from another script. The CLI at
the bottom is a quick self-test against logistic regression.
"""

from __future__ import annotations

import random
import numpy as np
from deap import base, creator, tools, algorithms

from evaluate import cv_evaluate, make_splits


# Default GA hyperparameters (override per model via run_ga kwargs).
DEFAULTS = dict(
    population_size=120,
    n_generations=70,
    crossover_prob=0.45,
    mutation_prob=0.5,
    flip_prob_on=0.15,   # prob of dropping a currently-selected feature (1 -> 0)
    flip_prob_off=0.1, # prob of adding an unselected feature (0 -> 1)
    init_density=0.1,
    c_values=[0.01, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.0],
    c_mut_prob=0.2,     # probability of resampling C during mutation
)

RF_CONFIGS = [
    {"n_estimators": 100, "max_depth": None,  "max_features": "sqrt",  "min_samples_split": 2},
    {"n_estimators": 100, "max_depth": 10,    "max_features": "sqrt",  "min_samples_split": 5},
    {"n_estimators": 300, "max_depth": None,  "max_features": "sqrt",  "min_samples_split": 2},
    {"n_estimators": 300, "max_depth": 10,    "max_features": "sqrt",  "min_samples_split": 5},
    {"n_estimators": 300, "max_depth": 20,    "max_features": "log2",  "min_samples_split": 2},
    {"n_estimators": 500, "max_depth": None,  "max_features": "log2",  "min_samples_split": 2},
    {"n_estimators": 500, "max_depth": 20,    "max_features": "sqrt",  "min_samples_split": 10},
    {"n_estimators": 500, "max_depth": None,  "max_features": "log2",  "min_samples_split": 5},
]


DEFAULTS_RF = dict(
    population_size=120,
    n_generations=70,
    crossover_prob=0.45,
    mutation_prob=0.5,
    flip_prob_on=0.15,
    flip_prob_off=0.1,
    init_density=0.1,
    rf_configs=RF_CONFIGS,
    rf_mut_prob=0.2,
)


def _ensure_creator_classes():
    """DEAP's creator module rebinds class names globally; guard re-registration."""
    if not hasattr(creator, "FitnessMultiFS"):
        # weights=(1.0, -1.0): maximise accuracy, minimise feature count
        creator.create("FitnessMultiFS", base.Fitness, weights=(1.0, -1.0))
    if not hasattr(creator, "IndividualFS"):
        creator.create("IndividualFS", list, fitness=creator.FitnessMultiFS)


def _bit_with_density(p):
    return 1 if random.random() < p else 0


def _init_individual(icls, n_features, n_c, density):
    """Individual = n_features bits (feature mask) + 1 integer (C index)."""
    bits = [_bit_with_density(density) for _ in range(n_features)]
    return icls(bits + [random.randrange(n_c)])


def _mutate_mixed(individual, p_on_to_off, p_off_to_on, n_c, c_mut_prob):
    """Asymmetric bit-flip on feature genes; random resample on the C gene."""
    for i in range(len(individual) - 1):
        if random.random() < (p_on_to_off if individual[i] else p_off_to_on):
            individual[i] = 1 - individual[i]
    if random.random() < c_mut_prob:
        individual[-1] = random.randrange(n_c)
    return (individual,)


def _build_toolbox(n_features, n_c, fitness_fn, hp):
    _ensure_creator_classes()
    toolbox = base.Toolbox()
    toolbox.register("individual", _init_individual, creator.IndividualFS,
                     n_features, n_c, hp["init_density"])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fitness_fn)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", _mutate_mixed,
                     p_on_to_off=hp["flip_prob_on"],
                     p_off_to_on=hp["flip_prob_off"],
                     n_c=n_c,
                     c_mut_prob=hp["c_mut_prob"])
    toolbox.register("select", tools.selNSGA2)
    return toolbox


def svm_run_ga(make_model, X, y, splits, *, seed=0, verbose=True, **hp_overrides):
    """Run the GA and return (selected_feature_indices, best_C, best_cv_accuracy).

    `make_model(feature_idx, C) -> estimator`
    """
    hp = {**DEFAULTS, **hp_overrides}
    random.seed(seed)
    np.random.seed(seed)

    n_features = X.shape[1]
    c_values = hp["c_values"]
    n_c = len(c_values)

    def fitness(individual):
        idx = np.flatnonzero(np.asarray(individual[:-1], dtype=bool))
        C = c_values[individual[-1]]
        if idx.size == 0:
            return (0.0, 0)
        accs, _, _, _ = cv_evaluate(
            lambda fi: make_model(fi, C), X, y, splits, feature_idx=idx
        )
        return (float(accs.mean()), int(idx.size))

    toolbox = _build_toolbox(n_features, n_c, fitness, hp)
    pop_size = hp["population_size"]
    population = toolbox.population(n=pop_size)

    stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats.register("mean", np.mean)
    stats.register("max", np.max)

    algorithms.eaMuPlusLambda(
        population, toolbox,
        mu=pop_size, lambda_=pop_size,
        cxpb=hp["crossover_prob"], mutpb=hp["mutation_prob"],
        ngen=hp["n_generations"], stats=stats, halloffame=None, verbose=verbose,
    )

    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    best = max(pareto_front, key=lambda ind: ind.fitness.values[0])
    selected = np.flatnonzero(np.asarray(best[:-1], dtype=bool))
    best_C = c_values[best[-1]]
    return selected, best_C, float(best.fitness.values[0])


def svm_run_ga_bc(make_model, X, y, splits, bc_feature_set, *, seed=0, verbose=True, **hp_overrides):
    """Identical to run_ga but with a 3rd objective: maximise KEGG BC feature overlap.

    bc_feature_set: set of feature indices (from select_bc_features) that overlap
    at least one KEGG breast-cancer gene. Used as a fast, no-HTTP proxy for BC
    gene coverage inside the fitness function.
    """
    hp = {**DEFAULTS, **hp_overrides}
    random.seed(seed)
    np.random.seed(seed)

    n_features = X.shape[1]
    c_values = hp["c_values"]
    n_c = len(c_values)
    bc_arr = np.array(sorted(bc_feature_set), dtype=int)

    if not hasattr(creator, "FitnessTriFS"):
        creator.create("FitnessTriFS", base.Fitness, weights=(1.0, -1.0, 1.0))
    if not hasattr(creator, "IndividualTriFS"):
        creator.create("IndividualTriFS", list, fitness=creator.FitnessTriFS)

    def fitness(individual):
        idx = np.flatnonzero(np.asarray(individual[:-1], dtype=bool))
        C = c_values[individual[-1]]
        if idx.size == 0:
            return (0.0, 0, 0)
        accs, _, _, _ = cv_evaluate(
            lambda fi: make_model(fi, C), X, y, splits, feature_idx=idx
        )
        n_bc = int(np.isin(idx, bc_arr).sum())
        return (float(accs.mean()), int(idx.size), n_bc)

    toolbox = base.Toolbox()
    toolbox.register("individual", _init_individual, creator.IndividualTriFS,
                     n_features, n_c, hp["init_density"])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fitness)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", _mutate_mixed,
                     p_on_to_off=hp["flip_prob_on"],
                     p_off_to_on=hp["flip_prob_off"],
                     n_c=n_c, c_mut_prob=hp["c_mut_prob"])
    toolbox.register("select", tools.selNSGA2)

    pop_size = hp["population_size"]
    population = toolbox.population(n=pop_size)

    stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats.register("mean", np.mean)
    stats.register("max", np.max)

    algorithms.eaMuPlusLambda(
        population, toolbox,
        mu=pop_size, lambda_=pop_size,
        cxpb=hp["crossover_prob"], mutpb=hp["mutation_prob"],
        ngen=hp["n_generations"], stats=stats, halloffame=None, verbose=verbose,
    )

    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    best = max(pareto_front, key=lambda ind: ind.fitness.values[0])
    selected = np.flatnonzero(np.asarray(best[:-1], dtype=bool))
    best_C = c_values[best[-1]]
    return selected, best_C, float(best.fitness.values[0])

def rf_run_ga(make_model, X, y, splits, *, seed=0, verbose=True, **hp_overrides):
    """Run the GA and return (selected_feature_indices, best_cv_accuracy).
    """
    hp = {**DEFAULTS_RF, **hp_overrides}
    random.seed(seed)
    np.random.seed(seed)

    n_features = X.shape[1]
    hp_configs = hp["rf_configs"]
    n_confs = len(hp_configs)

    def fitness(individual):
        idx = np.flatnonzero(np.asarray(individual[:-1], dtype=bool))
        hp_config = RF_CONFIGS[individual[-1]]
        if idx.size == 0:
            return (0.0, 0)
        accs, _, _, _ = cv_evaluate(
            lambda fi: make_model(fi, hp_config), X, y, splits, feature_idx=idx
        )
        return (float(accs.mean()), int(idx.size))

    toolbox = _build_toolbox(n_features, n_confs, fitness, {**hp, "c_mut_prob": hp["rf_mut_prob"]})
    pop_size = hp["population_size"]
    population = toolbox.population(n=pop_size)

    stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats.register("mean", np.mean)
    stats.register("max", np.max)

    algorithms.eaMuPlusLambda(
        population, toolbox,
        mu=pop_size, lambda_=pop_size,
        cxpb=hp["crossover_prob"], mutpb=hp["mutation_prob"],
        ngen=hp["n_generations"], stats=stats, halloffame=None, verbose=verbose,
    )

    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    best = max(pareto_front, key=lambda ind: ind.fitness.values[0])
    selected = np.flatnonzero(np.asarray(best[:-1], dtype=bool))
    best_cfg = RF_CONFIGS[best[-1]]
    return selected, best_cfg, float(best.fitness.values[0])

def rf_run_ga_bc(make_model, X, y, splits, bc_feature_set, *, seed=0, verbose=True, **hp_overrides):
    """Run the GA and return (selected_feature_indices, best_cv_accuracy).
    """
    hp = {**DEFAULTS_RF, **hp_overrides}
    random.seed(seed)
    np.random.seed(seed)

    n_features = X.shape[1]
    hp_configs = hp["rf_configs"]
    n_confs = len(hp_configs)
    bc_arr = np.array(sorted(bc_feature_set), dtype=int)

    def fitness(individual):
        idx = np.flatnonzero(np.asarray(individual[:-1], dtype=bool))
        hp_config = RF_CONFIGS[individual[-1]]
        if idx.size == 0:
            return (0.0, 0, 0)
        accs, _, _, _ = cv_evaluate(
            lambda fi: make_model(fi, hp_config), X, y, splits, feature_idx=idx
        )
        n_bc = int(np.isin(idx, bc_arr).sum())
        return (float(accs.mean()), int(idx.size), n_bc)

    if not hasattr(creator, "FitnessTriFS"):
        creator.create("FitnessTriFS", base.Fitness, weights=(1.0, -1.0, 1.0))
    if not hasattr(creator, "IndividualTriFS"):
        creator.create("IndividualTriFS", list, fitness=creator.FitnessTriFS)

    toolbox = base.Toolbox()
    toolbox.register("individual", _init_individual, creator.IndividualTriFS,
                     n_features, n_confs, hp["init_density"])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fitness)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", _mutate_mixed,
                     p_on_to_off=hp["flip_prob_on"],
                     p_off_to_on=hp["flip_prob_off"],
                     n_c=n_confs, c_mut_prob=hp["rf_mut_prob"])
    toolbox.register("select", tools.selNSGA2)

    pop_size = hp["population_size"]
    population = toolbox.population(n=pop_size)

    stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats.register("mean", np.mean)
    stats.register("max", np.max)

    algorithms.eaMuPlusLambda(
        population, toolbox,
        mu=pop_size, lambda_=pop_size,
        cxpb=hp["crossover_prob"], mutpb=hp["mutation_prob"],
        ngen=hp["n_generations"], stats=stats, halloffame=None, verbose=verbose,
    )

    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    best = max(pareto_front, key=lambda ind: ind.fitness.values[0])
    selected = np.flatnonzero(np.asarray(best[:-1], dtype=bool))
    best_cfg = RF_CONFIGS[best[-1]]
    return selected, best_cfg, float(best.fitness.values[0])