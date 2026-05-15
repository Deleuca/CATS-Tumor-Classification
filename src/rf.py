
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
import numpy as np
import pandas as pd
from gene_selection import select_bc_features
from feature_selection import rf_run_ga, rf_run_ga_bc
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


def train_kegg_rf(X, y, genes_out="selected_bc_genes.txt"):
    """SVM trained on KEGG-curated BC gene features with GridSearchCV for C."""
    keep, bc_genes = select_bc_features(X.columns)
    with open(genes_out, "w") as f:
        f.write("\n".join(bc_genes) + "\n")
    print(f"KEGG filter: {len(keep)} features covering {len(bc_genes)} BC genes")

    param_grid = {
      "randomforestclassifier__n_estimators": [100, 300, 500],
      "randomforestclassifier__max_depth": [None, 5, 10, 20],
      "randomforestclassifier__max_features": ["sqrt", "log2"],
      "randomforestclassifier__min_samples_split": [2, 5, 10],
    }

    X_kegg = np.asarray(X)[:, keep]
    pipeline = make_pipeline(RandomForestClassifier(random_state = 42))
    gs = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    gs.fit(X_kegg, np.asarray(y))
    print(f"KEGG RF with CV acc={gs.best_score_:.3f}")
    return gs.best_estimator_, keep, bc_genes


def train_ga_rf(X, y, genes_out="ga_selected_genes.txt", **ga_kwargs):
    """GA-based joint feature selection and C optimisation over all features."""
    def make_model(_, cfg):
        return make_pipeline(RandomForestClassifier(**cfg, random_state=42, n_jobs=-1))


    splits = make_splits(np.asarray(y))
    selected, best_C, best_acc = rf_run_ga(make_model, np.asarray(X), np.asarray(y), splits, **ga_kwargs)

    _, genes = select_bc_features(selected)
    with open(genes_out, "w") as f:
        f.write("\n".join(genes) + "\n")

    print(f"GA selected {len(selected)} features, {len(genes)} KEGG BC genes, C={best_C}, CV acc={best_acc:.3f}")
    final = make_pipeline(RandomForestClassifier(random_state=42))
    final.fit(np.asarray(X)[:, selected], np.asarray(y))
    return final, selected, best_C


def train_kegg_ga_rf(X, y, keep, genes_out="kegg_ga_selected_genes.txt", **ga_kwargs):
    """GA with 3-objective fitness: accuracy, sparsity, KEGG BC feature overlap."""
    def make_model(_, C):
        return make_pipeline(StandardScaler(), LinearSVC(C=C, max_iter=5000, dual="auto"))

    splits = make_splits(np.asarray(y))
    selected, best_C, best_acc = run_ga_bc(make_model, np.asarray(X), np.asarray(y), splits, set(keep), **ga_kwargs)

    _, genes = select_bc_features(selected)
    with open(genes_out, "w") as f:
        f.write("\n".join(genes) + "\n")

    print(f"KEGG-GA selected {len(selected)} features, {len(genes)} KEGG BC genes, C={best_C}, CV acc={best_acc:.3f}")
    final = make_pipeline(StandardScaler(), LinearSVC(C=best_C, max_iter=5000, dual="auto"))
    final.fit(np.asarray(X)[:, selected], np.asarray(y))
    return final, selected, best_C
