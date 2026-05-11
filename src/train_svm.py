"""Train and persist the final SVM model.

Pipeline:
  1. Short hyperparameter grid search (C x class_weight) on all 2834 features,
     stratified 5-fold CV.
  2. GA feature selection using the best HP, same CV split.
  3. Refit StandardScaler + LinearSVC pipeline on the FULL data with the
     selected features.
  4. Pickle the trained pipeline + selection metadata to `model.pkl`.
  5. Write the selected probes (with Chromosome/Start/End) to
     `selected_features.csv`.

The honest generalization estimate for this configuration is the nested-CV
number reported in results.txt (sklearn LinearSVC at ~0.76 acc).

Usage (from repo root):
    python src/train_svm.py
"""

from __future__ import annotations

import sys
import time
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from svm import read_data
from evaluate import make_splits, cv_evaluate
from feature_selection import run_ga


GRID = [
    dict(C=C, class_weight=cw)
    for C in (0.01, 0.1, 1.0, 10.0)
    for cw in (None, "balanced")
]

MODEL_PATH = "model.pkl"
FEATURES_PATH = "selected_features.csv"


def make_factory(hp):
    """Returns a (feature_idx) -> Pipeline factory matching evaluate.cv_evaluate."""
    def factory(_idx):
        return make_pipeline(
            StandardScaler(),
            LinearSVC(C=hp["C"], class_weight=hp["class_weight"],
                      max_iter=5000, random_state=42, dual="auto"),
        )
    return factory


def main():
    X_df, y_s, chrom = read_data()
    X = X_df.to_numpy()
    y = y_s.to_numpy()
    classes = np.unique(y)
    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {classes.tolist()}\n")

    splits = make_splits(y)

    # ---- 1. Grid search ----
    print(f"--- HP grid search ({len(GRID)} configs, 5-fold stratified CV) ---")
    print(f"{'C':>6s}  {'class_weight':>14s}  {'accuracy':<20s}")
    print("-" * 46)
    best_acc, best_hp = -np.inf, None
    for hp in GRID:
        t0 = time.time()
        accs, _, _, _ = cv_evaluate(make_factory(hp), X, y, splits)
        dt = time.time() - t0
        marker = ""
        if accs.mean() > best_acc:
            best_acc, best_hp = accs.mean(), hp
            marker = "  <- best so far"
        print(f"{hp['C']:>6.2f}  {str(hp['class_weight']):>14s}  "
              f"{accs.mean():.3f} +/- {accs.std():.3f}  ({dt:.1f}s){marker}")
    print(f"\nBest HP: C={best_hp['C']}, class_weight={best_hp['class_weight']}  "
          f"(CV acc = {best_acc:.3f})\n")

    # ---- 2. GA feature selection with best HP ----
    print(f"--- GA feature selection (pop=50, gen=40, best HP) ---")
    t0 = time.time()
    selected, ga_best = run_ga(
        make_factory(best_hp), X, y, splits,
        population_size=50, n_generations=40,
        verbose=False, seed=0,
    )
    ga_dt = time.time() - t0
    print(f"GA selected {len(selected)} / {X.shape[1]} features  "
          f"(GA-CV acc {ga_best:.3f},  {ga_dt:.0f}s)\n")

    # ---- 3. Final fit on full data with selected features ----
    final_pipeline = make_factory(best_hp)(selected)
    final_pipeline.fit(X[:, selected], y)

    # Sanity CV on the selected set (optimistic; only for reporting)
    accs, f1s, conf, cls = cv_evaluate(
        make_factory(best_hp), X, y, splits, feature_idx=selected
    )
    print(f"Final pipeline 5-fold CV (selected features, same splits):")
    print(f"  acc      = {accs.mean():.3f} +/- {accs.std():.3f}")
    print(f"  macro-F1 = {f1s.mean():.3f} +/- {f1s.std():.3f}")

    # ---- 4. Persist model ----
    bundle = {
        "pipeline": final_pipeline,
        "selected_features": np.asarray(selected, dtype=int),
        "hyperparameters": best_hp,
        "classes_": cls,
        "cv_accuracy": float(accs.mean()),
        "cv_accuracy_std": float(accs.std()),
        "cv_macro_f1": float(f1s.mean()),
        "ga_population": 50,
        "ga_generations": 40,
        "model_name": "sklearn LinearSVC (GA-selected, HP-tuned)",
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nPickled model -> {MODEL_PATH}  (keys: {list(bundle)})")

    # ---- 5. Selected features as CSV (with probe metadata) ----
    call = pd.read_csv("B4TM_CATS_training_data/Train_call.tsv", sep="\t")
    feat_meta = call.iloc[selected][["Chromosome", "Start", "End", "Nclone"]].copy()
    feat_meta.insert(0, "feature_idx", selected)
    feat_meta.to_csv(FEATURES_PATH, index=False)
    print(f"Wrote selected probes -> {FEATURES_PATH}  ({len(selected)} rows)")

    # Per-chromosome breakdown for quick sanity check
    by_chrom = feat_meta.groupby("Chromosome").size().sort_index()
    print(f"\nSelected probes per chromosome:")
    for c, n in by_chrom.items():
        print(f"  chr {int(c):>2d}: {int(n):>4d}")


if __name__ == "__main__":
    main()
