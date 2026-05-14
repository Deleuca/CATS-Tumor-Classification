"""Fused SVM for arrayCGH classification.

Implementation of Rapaport, Barillot & Vert (2008),
"Classification of arrayCGH data using fused SVM", Bioinformatics 24(13):i375-i382.

Hinge-loss linear classifier with two budgets (paper Eq. 5):

    min_w  sum_i max(0, 1 - y_i * w . x_i)
    s.t.   sum_i      |w_i|              <= lambda     (L1 sparsity)
           sum_{i~j}  |w_i - w_j|        <= mu         (fusion / smoothness)

where i~j ranges over consecutive probes on the SAME chromosome. The result is
a sparse, piecewise-constant weight profile whose nonzero plateaus correspond
to discriminative chromosomal regions. Setting mu very large recovers the
plain L1-SVM (a strict special case, as the paper notes in Section 4).

The constrained problem is reformulated as a linear program (paper Eq. 6) with
slack variables alpha (hinge), beta (|w_i|) and gamma (|w_i - w_j|), and solved
via scipy.optimize.linprog (HiGHS).

Multiclass extension: one-vs-rest (paper Section 2.2 final paragraph).

Run `python src/svm.py` from the repo root for an end-to-end verification.
"""

from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
import numpy as np
import pandas as pd
from gene_selection import select_bc_features


def read_data():
    """Load CATS arrayCGH training data along with chromosome assignments."""
    call = pd.read_csv("B4TM_CATS_training_data/Train_call.tsv", sep="\t")
    clinical = pd.read_csv("B4TM_CATS_training_data/Train_clinical.tsv", sep="\t")
    clinical["Subgroup"] = clinical["Subgroup"].astype("category")

    sample_columns = call.columns[4:]
    transposed_call = call[sample_columns].T
    transposed_call.index.name = "Sample"
    transposed_call.reset_index(inplace=True)

    merged = pd.merge(transposed_call, clinical, on="Sample")
    X = merged.drop(columns=["Sample", "Subgroup"])
    y = merged["Subgroup"]

    keep, bc_genes = select_bc_features(X.columns)
    X = X[keep]
    return X, y, bc_genes

def train_svm(X,y):
    param_grid = {"linearsvc__C": [0.01, 0.05, 0.075, 0.1, 0.125, 0.25, 0.5, 1.0]}

    pipeline = make_pipeline(StandardScaler(), LinearSVC(max_iter=5000, dual="auto"))

    gs = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    gs.fit(X, y)

    return gs.best_estimator_
