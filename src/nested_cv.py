"""Nested cross-validation with GA feature selection in the inner loop.

`compare_models.py` reuses the same folds for GA fitness and for the final
report -- the GA partially overfits those folds, so the numbers are optimistic.
This script runs an honest nested CV: outer 5-fold for scoring, inner 5-fold
for the GA's fitness. The outer test fold is never touched during selection.

Same GA settings per model as compare_models.py. Reduced settings for the
fused SVM since each fitness eval is a 3-class OvR LP solve.

Usage (from repo root):
    python src/nested_cv.py                      # default: knn,lr,rf,linsvm
    python src/nested_cv.py --models lr,linsvm   # subset
    python src/nested_cv.py --models all         # everything incl. fused-SVM
"""

from __future__ import annotations

import argparse
import sys
import time
import numpy as np

sys.path.insert(0, "src")

from evaluate import nested_cv
from feature_selection import run_ga
from svm import read_data
from compare_models import (
    factory_knn, factory_lr, factory_rf,
    factory_sklearn_svm, make_factory_svm,
)


def _model_specs(chrom):
    """Return the full set of models keyed by short CLI names."""
    return {
        "knn":    ("KNN (k=5)",            factory_knn,
                   dict(population_size=50, n_generations=40)),
        "lr":     ("LogisticRegression",   factory_lr,
                   dict(population_size=50, n_generations=40)),
        "rf":     ("RandomForest (n=200)", factory_rf,
                   dict(population_size=30, n_generations=20)),
        "linsvm": ("sklearn LinearSVC",    factory_sklearn_svm,
                   dict(population_size=50, n_generations=40)),
        "fused":  ("Fused-SVM (L1, mu=inf)",
                   make_factory_svm(chrom, lambda_=50.0, mu=1e6),
                   dict(population_size=15, n_generations=8)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", default="knn,lr,rf,linsvm",
        help="comma-separated subset of {knn,lr,rf,linsvm,fused} or 'all'",
    )
    args = parser.parse_args()

    X_df, y_s, chrom = read_data()
    X = X_df.to_numpy()
    y = y_s.to_numpy()
    classes, counts = np.unique(y, return_counts=True)

    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{int(np.unique(chrom).size)} chromosomes")
    print(f"Classes: {dict(zip(classes.tolist(), counts.tolist()))}")
    print(f"\nOuter 5-fold StratifiedKFold(shuffle=True, random_state=0)")
    print(f"Inner 5-fold per outer fold, GA fitness = inner-CV mean accuracy")
    print(f"Selected subset is refit on the FULL outer-train and scored on "
          f"the held-out outer-test fold.\n")

    specs = _model_specs(chrom)
    keys = list(specs.keys()) if args.models.strip() == "all" \
        else [k.strip() for k in args.models.split(",")]
    unknown = [k for k in keys if k not in specs]
    if unknown:
        parser.error(f"unknown model(s): {unknown}. choices: {list(specs)}")

    summary, details = [], {}

    for key in keys:
        label, factory, ga_hp = specs[key]
        print("=" * 78)
        print(f"Nested CV: {label}    GA hp = {ga_hp}")
        print("=" * 78)
        t0 = time.time()
        out = nested_cv(factory, X, y, run_ga, ga_hp=ga_hp, verbose=True)
        dt = time.time() - t0
        details[label] = out
        summary.append((label, out["accs"], out["f1s"], out["n_features"], dt))
        print(f"\n  >>> {label}: outer-CV acc = "
              f"{out['accs'].mean():.3f} +/- {out['accs'].std():.3f}    "
              f"macro-F1 = {out['f1s'].mean():.3f} +/- {out['f1s'].std():.3f}    "
              f"avg features = {int(out['n_features'].mean())}    "
              f"({dt:.0f}s)\n")

    print("=" * 78)
    print("Nested-CV summary -- sorted by mean outer-test accuracy")
    print("=" * 78)
    header = (f"{'model':<26s}  {'accuracy':<18s}  {'macro-F1':<18s}  "
              f"{'avg #feats':>10s}  {'time':>7s}")
    print(header); print("-" * len(header))
    summary.sort(key=lambda r: -r[1].mean())
    for label, accs, f1s, n, dt in summary:
        print(f"{label:<26s}  "
              f"{accs.mean():.3f} +/- {accs.std():.3f}    "
              f"{f1s.mean():.3f} +/- {f1s.std():.3f}    "
              f"{int(n.mean()):>10d}  {dt:>6.0f}s")

    best_label = summary[0][0]
    out = details[best_label]
    print(f"\nBest model (honest nested CV): {best_label}")
    print(f"  per-fold accuracy : {np.round(out['accs'], 3).tolist()}")
    print(f"  per-fold #features: {out['n_features'].tolist()}")

    print(f"\nAggregated outer-test confusion matrix -- {best_label}")
    cls = out["classes"]
    print(f"  rows = true, cols = predicted; class order = {cls.tolist()}")
    width = max(len(c) for c in cls) + 2
    print("  " + "".join(f"{c:>{width}s}" for c in [""] + list(cls)))
    for i, c in enumerate(cls):
        print("  " + f"{c:>{width}s}"
              + "".join(f"{int(v):>{width}d}" for v in out["confusion"][i]))


if __name__ == "__main__":
    main()
