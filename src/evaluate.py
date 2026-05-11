"""Unified cross-validation harness.

Mirrors the scheme already used by svm_cv.py: StratifiedKFold(5, shuffle=True,
random_state=0), reporting per-fold accuracy and macro-F1 plus an aggregated
confusion matrix. Every model in this project should be compared through this
function so the comparison is fair (identical splits, identical metrics).

A model is supplied as a "factory" function: `make_model(feature_idx)` returns
a fresh, fit-ready estimator that will be trained on X[:, feature_idx]. The
factory pattern lets the fused SVM thread its per-feature chromosome IDs
through to the constructor when the GA selects a subset of probes.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def make_splits(y, n_splits=5, random_state=0):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(np.zeros(len(y)), y))


def cv_evaluate(make_model, X, y, splits, feature_idx=None):
    """Run CV and return (accs, f1s, aggregated confusion matrix, classes).

    Parameters
    ----------
    make_model : callable(feature_idx) -> estimator
        Returns an unfit estimator. `feature_idx` is the array of column indices
        the estimator will see — useful for estimators that need per-feature
        side info (e.g. fused SVM needs chromosome IDs aligned to the subset).
    X, y : arrays
    splits : list of (train_idx, test_idx) — produced by make_splits
    feature_idx : array-like of int or None
        Column subset to train/test on. None means all columns.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if feature_idx is None:
        feature_idx = np.arange(X.shape[1])
    else:
        feature_idx = np.asarray(feature_idx, dtype=int)

    classes = np.unique(y)
    Xs = X[:, feature_idx]

    accs, f1s, conf = [], [], None
    for tr, te in splits:
        clf = make_model(feature_idx)
        clf.fit(Xs[tr], y[tr])
        yhat = clf.predict(Xs[te])
        accs.append(accuracy_score(y[te], yhat))
        f1s.append(f1_score(y[te], yhat, average="macro"))
        cm = confusion_matrix(y[te], yhat, labels=classes)
        conf = cm if conf is None else conf + cm
    return np.array(accs), np.array(f1s), conf, classes


def nested_cv(make_model, X, y, ga_run, *,
              outer_n_splits=5, inner_n_splits=5,
              outer_random_state=0, inner_random_state_base=0,
              ga_seed_base=0, ga_hp=None, verbose=True):
    """Nested CV: GA in the inner loop, honest scoring on the outer test fold.

    For each outer fold:
      1. Build inner splits over the outer-train rows.
      2. Run `ga_run(make_model, X_tr, y_tr, inner_splits, seed, **ga_hp)` to
         pick a feature subset using only inner CV (the outer test fold is
         never seen during selection).
      3. Refit `make_model(selected)` on the full outer-train.
      4. Score on the held-out outer-test.

    Returns a dict with per-fold accuracy/macro-F1, the selected feature counts,
    and the aggregated confusion matrix.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    ga_hp = ga_hp or {}
    classes = np.unique(y)

    outer_skf = StratifiedKFold(n_splits=outer_n_splits, shuffle=True,
                                random_state=outer_random_state)
    outer_splits = list(outer_skf.split(np.zeros(len(y)), y))

    fold_accs, fold_f1s, fold_n = [], [], []
    fold_selected = []
    conf = None

    for k, (tr, te) in enumerate(outer_splits):
        X_tr, y_tr = X[tr], y[tr]
        X_te, y_te = X[te], y[te]

        inner_skf = StratifiedKFold(n_splits=inner_n_splits, shuffle=True,
                                    random_state=inner_random_state_base + k)
        inner_splits = list(inner_skf.split(np.zeros(len(y_tr)), y_tr))

        if verbose:
            print(f"  outer fold {k + 1}/{outer_n_splits}: running GA "
                  f"(inner {inner_n_splits}-fold)...")
        selected, ga_best = ga_run(make_model, X_tr, y_tr, inner_splits,
                                   seed=ga_seed_base + k, verbose=False, **ga_hp)
        if len(selected) == 0:
            selected = np.arange(X.shape[1])  # fallback — degenerate GA result

        clf = make_model(selected)
        clf.fit(X_tr[:, selected], y_tr)
        yhat = clf.predict(X_te[:, selected])

        acc = accuracy_score(y_te, yhat)
        f1 = f1_score(y_te, yhat, average="macro")
        cm = confusion_matrix(y_te, yhat, labels=classes)

        fold_accs.append(acc)
        fold_f1s.append(f1)
        fold_n.append(int(len(selected)))
        fold_selected.append(np.asarray(selected, dtype=int))
        conf = cm if conf is None else conf + cm

        if verbose:
            print(f"    inner-GA best={ga_best:.3f}  features={len(selected)}  "
                  f"outer-test acc={acc:.3f}  macro-F1={f1:.3f}")

    return dict(
        accs=np.array(fold_accs),
        f1s=np.array(fold_f1s),
        n_features=np.array(fold_n),
        selected=fold_selected,
        confusion=conf,
        classes=classes,
    )


def format_row(label, accs, f1s, n_features=None, elapsed=None):
    parts = [
        f"{label:<32s}",
        f"{accs.mean():.3f} +/- {accs.std():.3f}",
        f"{f1s.mean():.3f} +/- {f1s.std():.3f}",
    ]
    if n_features is not None:
        parts.append(f"feats={n_features:>5d}")
    if elapsed is not None:
        parts.append(f"{elapsed:>5.0f}s")
    return "  ".join(parts)
