"""Cross-validation evaluation of the fused SVM on the CATS arrayCGH data.

Runs stratified 5-fold CV across a small (lambda, mu) grid, reporting mean +/-
std accuracy and macro-F1 per configuration. mu = 1e6 acts as a plain L1-SVM
baseline (paper Section 4 notes that mu > 2*lambda effectively turns fusion off).

This is a *hyperparameter sweep*, not unbiased generalization estimation: the
same CV folds are reused across configs, so picking the best row over-fits the
fold split slightly. Use nested CV if you need an unbiased estimate of the
selected model.

Usage (from repo root):
    python src/svm_cv.py
"""

import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from svm import read_data, FusedSVMOvR


def cv_evaluate(X, y, chrom, lambda_, mu, splits):
    """Return per-fold accuracy and macro-F1 plus an aggregate confusion matrix."""
    accs, f1s, conf = [], [], None
    classes = np.unique(y)
    for tr, te in splits:
        clf = FusedSVMOvR(lambda_=lambda_, mu=mu, chromosomes=chrom).fit(X[tr], y[tr])
        yhat = clf.predict(X[te])
        accs.append(accuracy_score(y[te], yhat))
        f1s.append(f1_score(y[te], yhat, average="macro"))
        cm = confusion_matrix(y[te], yhat, labels=classes)
        conf = cm if conf is None else conf + cm
    return np.array(accs), np.array(f1s), conf, classes


def main():
    X, y, chrom = read_data()
    X = X.to_numpy()
    y = y.to_numpy()
    classes, counts = np.unique(y, return_counts=True)

    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features (chromosomes: "
          f"{int(np.unique(chrom).size)})")
    print(f"Classes: {dict(zip(classes.tolist(), counts.tolist()))}\n")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    splits = list(skf.split(X, y))

    # Each config requires 5 folds * 3 OvR classes = 15 LP solves.
    configs = [
        (50.0, 1e6, "L1-SVM    (mu=inf)"),
        (50.0, 10.0, "fused     (mu=10)"),
        (50.0,  5.0, "fused     (mu=5) "),
        (50.0,  1.0, "fused     (mu=1) "),
        (20.0,  5.0, "fused     (l=20,mu=5)"),
        (100.0, 5.0, "fused     (l=100,mu=5)"),
    ]

    print(f"5-fold stratified CV  (each row = mean +/- std over folds)")
    header = f"{'config':<24s}  {'accuracy':<16s}  {'macro-F1':<16s}  {'time':>6s}"
    print(header)
    print("-" * len(header))

    results = []
    for lambda_, mu, label in configs:
        t0 = time.time()
        accs, f1s, conf, cls = cv_evaluate(X, y, chrom, lambda_, mu, splits)
        dt = time.time() - t0
        print(f"{label:<24s}  {accs.mean():.3f} +/- {accs.std():.3f}    "
              f"{f1s.mean():.3f} +/- {f1s.std():.3f}    {dt:>5.0f}s")
        results.append((label, accs, f1s, conf, cls))

    # Aggregate confusion matrix for the best-by-accuracy config (informational).
    best = max(results, key=lambda r: r[1].mean())
    label, accs, f1s, conf, cls = best
    print(f"\nAggregated CV confusion matrix for best config: {label.strip()}")
    print(f"  rows = true, cols = predicted; class order = {cls.tolist()}")
    width = max(len(c) for c in cls) + 2
    print("  " + "".join(f"{c:>{width}s}" for c in [""] + list(cls)))
    for i, c in enumerate(cls):
        print("  " + f"{c:>{width}s}" + "".join(f"{int(v):>{width}d}" for v in conf[i]))


if __name__ == "__main__":
    main()
