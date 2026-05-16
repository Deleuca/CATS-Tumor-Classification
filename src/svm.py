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
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from gene_selection import select_bc_features, genes_for_features
from feature_selection import svm_run_ga, svm_run_ga_bc
from evaluate import make_splits


def read_data():
    """Load CATS arrayCGH training data."""
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
    return X, y


def train_kegg_svm(X, y, c_values=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0], genes_out="selected_bc_genes.txt"):
    """SVM trained on KEGG-curated BC gene features with GridSearchCV for C."""
    keep, bc_genes = select_bc_features(X.columns)
    with open(genes_out, "w") as f:
        f.write("\n".join(bc_genes) + "\n")
    print(f"KEGG filter: {len(keep)} features covering {len(bc_genes)} BC genes")

    X_kegg = np.asarray(X)[:, keep]
    pipeline = make_pipeline(StandardScaler(), LinearSVC(max_iter=5000, dual="auto"))
    gs = GridSearchCV(pipeline, {"linearsvc__C": c_values}, cv=5, scoring="accuracy", n_jobs=-1)
    gs.fit(X_kegg, np.asarray(y))
    print(f"KEGG SVM best C={gs.best_params_['linearsvc__C']}, CV acc={gs.best_score_:.3f}")
    return gs.best_estimator_, keep, bc_genes


def train_ga_svm(X, y, genes_out="svm_ga_selected_genes.txt", **ga_kwargs):
    """GA-based joint feature selection and C optimisation over all features."""
    def make_model(_, C):
        return make_pipeline(StandardScaler(), LinearSVC(C=C, max_iter=5000, dual="auto"))

    splits = make_splits(np.asarray(y))
    selected, best_C, best_acc, front = svm_run_ga(make_model, np.asarray(X), np.asarray(y), splits, **ga_kwargs)

    genes = genes_for_features(selected)
    with open(genes_out, "w") as f:
        f.write("\n".join(genes) + "\n")

    print(f"GA selected {len(selected)} features, {len(genes)} genes, C={best_C}, CV acc={best_acc:.3f}")
    final = make_pipeline(StandardScaler(), LinearSVC(C=best_C, max_iter=5000, dual="auto"))
    final.fit(np.asarray(X)[:, selected], np.asarray(y))
    return final, selected, best_C, front


def train_kegg_ga_svm(X, y, keep, genes_out="svm_kegg_ga_selected_genes.txt", **ga_kwargs):
    """GA with 3-objective fitness: accuracy, sparsity, KEGG BC feature overlap."""
    def make_model(_, C):
        return make_pipeline(StandardScaler(), LinearSVC(C=C, max_iter=5000, dual="auto"))

    splits = make_splits(np.asarray(y))
    selected, best_C, best_acc, front = svm_run_ga_bc(make_model, np.asarray(X), np.asarray(y), splits, set(keep), **ga_kwargs)

    genes = genes_for_features(selected)
    with open(genes_out, "w") as f:
        f.write("\n".join(genes) + "\n")

    print(f"KEGG-GA selected {len(selected)} features, {len(genes)} genes, C={best_C}, CV acc={best_acc:.3f}")
    final = make_pipeline(StandardScaler(), LinearSVC(C=best_C, max_iter=5000, dual="auto"))
    final.fit(np.asarray(X)[:, selected], np.asarray(y))
    return final, selected, best_C, front
