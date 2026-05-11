"""Fair head-to-head comparison of every classifier in this repo.

For each model:
  1. Baseline CV (all features)
  2. GA feature selection
  3. CV again on the GA-selected feature subset

Every CV uses the same splits (src/evaluate.make_splits, the scheme used in
svm_cv.py). The final table makes the GA's effect visible per-model and lets
you pick the best overall configuration.

GA on the fused SVM uses reduced settings (per the user's preference): each
fitness eval is a 3-class OvR LP solve that takes ~30s, so full-strength GA
would take many hours. Reduced pop/gen keeps the comparison feasible.

Usage (from repo root):
    python src/compare_models.py
"""

from __future__ import annotations

import sys
import time
import numpy as np

sys.path.insert(0, "src")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import cv_evaluate, make_splits, format_row
from feature_selection import run_ga
from svm import read_data, FusedSVMOvR


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------
def factory_knn(_idx):
    return KNeighborsClassifier(n_neighbors=5)


def factory_lr(_idx):
    return LogisticRegression(max_iter=1000, random_state=42)


def factory_rf(_idx):
    return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)


def factory_sklearn_svm(_idx):
    """sklearn LinearSVC with input scaling (linear SVM, L2 penalty, squared-hinge
    loss by default). Scaling is important: arrayCGH values are discrete and
    LinearSVC's solver behaves much better on scaled inputs.
    """
    return make_pipeline(
        StandardScaler(with_mean=True),
        LinearSVC(C=1.0, max_iter=5000, random_state=42, dual="auto"),
    )


def make_factory_svm(chrom_full, lambda_=50.0, mu=1e6):
    """Fused SVM needs the per-feature chromosome IDs aligned to the selected
    subset. Returns a closure suitable for evaluate.cv_evaluate.

    Default mu is large so this acts as the L1-SVM (best config from
    results.txt). Override lambda_/mu for fused configurations.
    """
    chrom_full = np.asarray(chrom_full)

    def factory(feature_idx):
        return FusedSVMOvR(lambda_=lambda_, mu=mu,
                           chromosomes=chrom_full[feature_idx])
    return factory


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def evaluate_one(label, factory, X, y, splits):
    t0 = time.time()
    accs, f1s, conf, classes = cv_evaluate(factory, X, y, splits)
    return accs, f1s, conf, classes, time.time() - t0


def ga_then_eval(label, factory, X, y, splits, **ga_hp):
    print(f"\n--- GA: {label} ---")
    t0 = time.time()
    selected, ga_best = run_ga(factory, X, y, splits, verbose=True, **ga_hp)
    ga_dt = time.time() - t0
    t1 = time.time()
    accs, f1s, conf, classes = cv_evaluate(factory, X, y, splits, feature_idx=selected)
    return selected, ga_best, ga_dt, accs, f1s, conf, classes, time.time() - t1


def main():
    X_df, y_s, chrom = read_data()
    X = X_df.to_numpy()
    y = y_s.to_numpy()
    classes, counts = np.unique(y, return_counts=True)

    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{int(np.unique(chrom).size)} chromosomes")
    print(f"Classes: {dict(zip(classes.tolist(), counts.tolist()))}")

    splits = make_splits(y)

    factory_svm = make_factory_svm(chrom, lambda_=50.0, mu=1e6)  # L1-SVM (best from earlier sweep)

    models = [
        ("KNN (k=5)",            factory_knn, dict(population_size=50, n_generations=40)),
        ("LogisticRegression",   factory_lr,  dict(population_size=50, n_generations=40)),
        ("RandomForest (200)",   factory_rf,  dict(population_size=30, n_generations=20)),
        ("Fused-SVM (L1, mu=inf)", factory_svm, dict(population_size=15, n_generations=8)),
    ]

    print("\n" + "=" * 78)
    print("Baseline CV (all features) — 5-fold stratified, shuffle, random_state=0")
    print("=" * 78)
    header = f"{'model':<32s}  {'accuracy':<16s}  {'macro-F1':<16s}  {'time':>6s}"
    print(header); print("-" * len(header))

    baseline = {}
    for label, factory, _ in models:
        accs, f1s, conf, cls, dt = evaluate_one(label, factory, X, y, splits)
        baseline[label] = (accs, f1s, conf, cls)
        print(format_row(label, accs, f1s, n_features=X.shape[1], elapsed=dt))

    print("\n" + "=" * 78)
    print("Genetic-algorithm feature selection, then CV on selected subset")
    print("=" * 78)
    ga_results = {}
    for label, factory, ga_hp in models:
        selected, ga_best, ga_dt, accs, f1s, conf, cls, cv_dt = ga_then_eval(
            label, factory, X, y, splits, **ga_hp
        )
        ga_results[label] = (selected, ga_best, accs, f1s, conf, cls)
        print(f"\n>>> {label}: GA selected {len(selected)} features "
              f"(GA={ga_dt:.0f}s, post-CV={cv_dt:.0f}s)")
        print(format_row(label + " (GA)", accs, f1s,
                         n_features=len(selected), elapsed=cv_dt))

    print("\n" + "=" * 78)
    print("Summary — baseline vs GA, sorted by GA accuracy")
    print("=" * 78)
    header = f"{'model':<32s}  {'baseline acc':<18s}  {'GA acc':<18s}  {'GA #feats':>10s}"
    print(header); print("-" * len(header))
    rows = []
    for label, _, _ in models:
        b_accs, _, _, _ = baseline[label]
        sel, _, g_accs, _, _, _ = ga_results[label]
        rows.append((label, b_accs, g_accs, len(sel)))
    rows.sort(key=lambda r: -r[2].mean())
    for label, b_accs, g_accs, n_sel in rows:
        print(f"{label:<32s}  "
              f"{b_accs.mean():.3f} +/- {b_accs.std():.3f}    "
              f"{g_accs.mean():.3f} +/- {g_accs.std():.3f}    "
              f"{n_sel:>10d}")

    best_label, b_accs, g_accs, n_sel = rows[0]
    print(f"\nBest model after GA: {best_label} "
          f"({g_accs.mean():.3f} +/- {g_accs.std():.3f} acc, "
          f"{n_sel} features)")

    # Aggregated CV confusion matrix for the best post-GA model.
    sel, _, accs, f1s, conf, cls = ga_results[best_label]
    print(f"\nAggregated CV confusion matrix — {best_label} (post-GA)")
    print(f"  rows = true, cols = predicted; class order = {cls.tolist()}")
    width = max(len(c) for c in cls) + 2
    print("  " + "".join(f"{c:>{width}s}" for c in [""] + list(cls)))
    for i, c in enumerate(cls):
        print("  " + f"{c:>{width}s}" + "".join(f"{int(v):>{width}d}" for v in conf[i]))


if __name__ == "__main__":
    main()
