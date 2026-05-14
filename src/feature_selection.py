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
    population_size=50,
    n_generations=40,
    crossover_prob=0.7,
    mutation_prob=0.2,
    flip_prob_on=0.8,   # prob of dropping a currently-selected feature (1 -> 0)
    flip_prob_off=0.1,  # prob of adding an unselected feature (0 -> 1)
    init_density=0.5,
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


def _asym_flip_bit(individual, p_on_to_off, p_off_to_on):
    """Asymmetric bit-flip mutation: selected features drop with p_on_to_off,
    unselected features activate with p_off_to_on."""
    for i, bit in enumerate(individual):
        if random.random() < (p_on_to_off if bit else p_off_to_on):
            individual[i] = 1 - bit
    return (individual,)


def _build_toolbox(n_features, fitness_fn, hp):
    _ensure_creator_classes()
    toolbox = base.Toolbox()
    toolbox.register("attr_bit", _bit_with_density, hp["init_density"])
    toolbox.register("individual", tools.initRepeat, creator.IndividualFS,
                     toolbox.attr_bit, n=n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fitness_fn)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", _asym_flip_bit,
                     p_on_to_off=hp["flip_prob_on"],
                     p_off_to_on=hp["flip_prob_off"])
    toolbox.register("select", tools.selNSGA2)
    return toolbox


def run_ga(make_model, X, y, splits, *, seed=0, verbose=True, **hp_overrides):
    """Run the GA for a given classifier factory and return the selected feature
    indices plus the GA's best CV accuracy.

    `make_model(feature_idx) -> estimator` matches the contract used by
    evaluate.cv_evaluate.
    """
    hp = {**DEFAULTS, **hp_overrides}
    random.seed(seed)
    np.random.seed(seed)

    n_features = X.shape[1]

    def fitness(individual):
        idx = np.flatnonzero(np.asarray(individual, dtype=bool))
        if idx.size == 0:
            return (0.0, 0)
        accs, _, _, _ = cv_evaluate(make_model, X, y, splits, feature_idx=idx)
        return (float(accs.mean()), int(idx.size))

    toolbox = _build_toolbox(n_features, fitness, hp)
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

    # Pick the highest-accuracy individual from the final Pareto front.
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    best = max(pareto_front, key=lambda ind: ind.fitness.values[0])
    selected = np.flatnonzero(np.asarray(best, dtype=bool))
    return selected, float(best.fitness.values[0])


# ---------------------------------------------------------------------------
# Self-test: GA on logistic regression
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from sklearn.linear_model import LogisticRegression
    from svm import read_data  # canonical loader (returns X, y, chromosomes)

    X_df, y_s, _ = read_data()
    X_arr = X_df.to_numpy()
    y_arr = y_s.to_numpy()

    splits = make_splits(y_arr)

    def make_lr(_idx):
        return LogisticRegression(max_iter=1000, random_state=42)

    # Smoke-test settings so this finishes in a couple of minutes.
    selected, best_acc = run_ga(
        make_lr, X_arr, y_arr, splits,
        population_size=20, n_generations=5, verbose=True,
    )
    print(f"\nSelected {len(selected)} features out of {X_arr.shape[1]}")
    print(f"Best CV accuracy (during GA): {best_acc:.4f}")
