import sys
sys.path.insert(0, "src")

from svm import *
from rf import train_ga_rf, train_kegg_ga_rf, KEGG_RF_PARAM_GRID
from gene_selection import select_bc_features
from evaluate import make_splits
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import json
import os

MODE     = sys.argv[1] if len(sys.argv) > 1 else "svm"
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else f"results_{MODE}.json"

os.makedirs("data", exist_ok=True)

X, y = read_data()
X_arr, y_arr = np.asarray(X), np.asarray(y)

# Compute KEGG feature mask once — it's coordinate-based, fold-independent
keep, bc_genes = select_bc_features(X.columns)
print(f"KEGG: {len(keep)} features covering {len(bc_genes)} BC genes")
with open("data/kegg_genes.txt", "w") as f:
    f.write("\n".join(bc_genes) + "\n")

splits = make_splits(y_arr)  # list of 5 (train_idx, test_idx) pairs
sets = []
for fold, (train_idx, test_idx) in enumerate(splits):
    X_train, X_test = X_arr[train_idx], X_arr[test_idx]
    y_train, y_test = y_arr[train_idx], y_arr[test_idx]
    sets.append((X_train, X_test, y_train, y_test))  # 0 mod 3 → GA
    sets.append((X_train, X_test, y_train, y_test))  # 1 mod 3 → KEGG
    sets.append((X_train, X_test, y_train, y_test))  # 2 mod 3 → KEGG-GA

ga_metrics = []
base_metrics = []
kegg_ga_metrics = []
ga_fronts = []
kegg_ga_fronts = []

rf_ga_metrics = []
rf_base_metrics = []
rf_kegg_ga_metrics = []
rf_ga_fronts = []
rf_kegg_ga_fronts = []

def svm():
    for i in range(15):
        X_train, X_test, y_train, y_test = sets[i]
        fold_num = i // 3 + 1

        if i % 3 == 0:
            print(f"[{i+1}/15] Fold {fold_num} — running GA SVM...")
            ga_model, selected, best_C, front = train_ga_svm(X_train, y_train, genes_out=f"data/svm_ga_genes_fold_{fold_num}.txt")
            y_pred = ga_model.predict(X_test[:, selected])
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            cm = confusion_matrix(y_test, y_pred)
            ga_metrics.append((acc, f1, cm))
            ga_fronts.append(front)
            with open(f"data/pareto_ga_fold_{fold_num}.json", "w") as f:
                json.dump(front, f, indent=2)
            print(f"Fold {fold_num} GA      — acc={acc:.3f}  macro-F1={f1:.3f}  features={len(selected)}  C={best_C}  pareto={len(front)}")

        elif i % 3 == 1:
            print(f"[{i+1}/15] Fold {fold_num} — running KEGG SVM...")
            gs = GridSearchCV(
                make_pipeline(StandardScaler(), LinearSVC(max_iter=5000, dual="auto")),
                {"linearsvc__C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]},
                cv=5, scoring="accuracy", n_jobs=-1,
            )
            gs.fit(X_train[:, keep], y_train)
            y_pred = gs.best_estimator_.predict(X_test[:, keep])
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            cm = confusion_matrix(y_test, y_pred)
            base_metrics.append((acc, f1, cm))
            print(f"Fold {fold_num} KEGG    — acc={acc:.3f}  macro-F1={f1:.3f}")

        else:
            print(f"[{i+1}/15] Fold {fold_num} — running KEGG-GA SVM...")
            kg_model, selected, best_C, front = train_kegg_ga_svm(X_train, y_train, keep, genes_out=f"data/svm_kegg_ga_genes_fold_{fold_num}.txt")
            y_pred = kg_model.predict(X_test[:, selected])
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            cm = confusion_matrix(y_test, y_pred)
            kegg_ga_metrics.append((acc, f1, cm))
            kegg_ga_fronts.append(front)
            with open(f"data/pareto_kegg_ga_fold_{fold_num}.json", "w") as f:
                json.dump(front, f, indent=2)
            print(f"Fold {fold_num} KEGG-GA — acc={acc:.3f}  macro-F1={f1:.3f}  features={len(selected)}  C={best_C}  pareto={len(front)}")

def rf():
    for i in range(15):
        X_train, X_test, y_train, y_test = sets[i]
        fold_num = i // 3 + 1

        if i % 3 == 0:
            print(f"[{i+1}/15] Fold {fold_num} — running GA RF...")
            rf_model, selected, best_cfg, front = train_ga_rf(X_train, y_train, genes_out=f"data/rf_ga_genes_fold_{fold_num}.txt")
            y_pred = rf_model.predict(X_test[:, selected])
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            cm = confusion_matrix(y_test, y_pred)
            rf_ga_metrics.append((acc, f1, cm))
            rf_ga_fronts.append(front)
            with open(f"data/pareto_rf_ga_fold_{fold_num}.json", "w") as f:
                json.dump(front, f, indent=2)
            print(f"Fold {fold_num} GA RF      — acc={acc:.3f}  macro-F1={f1:.3f}  features={len(selected)}  pareto={len(front)}")

        elif i % 3 == 1:
            print(f"[{i+1}/15] Fold {fold_num} — running KEGG RF...")
            gs = GridSearchCV(
                make_pipeline(RandomForestClassifier(random_state=42)),
                KEGG_RF_PARAM_GRID,
                cv=5, scoring="accuracy", n_jobs=-1,
            )
            gs.fit(X_train[:, keep], y_train)
            y_pred = gs.best_estimator_.predict(X_test[:, keep])
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            cm = confusion_matrix(y_test, y_pred)
            rf_base_metrics.append((acc, f1, cm))
            print(f"Fold {fold_num} KEGG RF    — acc={acc:.3f}  macro-F1={f1:.3f}")

        else:
            print(f"[{i+1}/15] Fold {fold_num} — running KEGG-GA RF...")
            kg_model, selected, best_cfg, front = train_kegg_ga_rf(X_train, y_train, keep, genes_out=f"data/rf_kegg_ga_genes_fold_{fold_num}.txt")
            y_pred = kg_model.predict(X_test[:, selected])
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            cm = confusion_matrix(y_test, y_pred)
            rf_kegg_ga_metrics.append((acc, f1, cm))
            rf_kegg_ga_fronts.append(front)
            with open(f"data/pareto_rf_kegg_ga_fold_{fold_num}.json", "w") as f:
                json.dump(front, f, indent=2)
            print(f"Fold {fold_num} KEGG-GA RF — acc={acc:.3f}  macro-F1={f1:.3f}  features={len(selected)}  pareto={len(front)}")


if MODE == "svm":
    svm()
else:
    rf()

if MODE == "svm":
    ga_accs      = np.array([m[0] for m in ga_metrics])
    ga_f1s       = np.array([m[1] for m in ga_metrics])
    kegg_accs    = np.array([m[0] for m in base_metrics])
    kegg_f1s     = np.array([m[1] for m in base_metrics])
    kegg_ga_accs = np.array([m[0] for m in kegg_ga_metrics])
    kegg_ga_f1s  = np.array([m[1] for m in kegg_ga_metrics])

    print("\n" + "=" * 65)
    print(f"{'':12s}  {'accuracy':>20s}  {'macro-F1':>20s}")
    print(f"{'GA SVM':12s}  {ga_accs.mean():.3f} +/- {ga_accs.std():.3f}  {ga_f1s.mean():.3f} +/- {ga_f1s.std():.3f}")
    print(f"{'KEGG SVM':12s}  {kegg_accs.mean():.3f} +/- {kegg_accs.std():.3f}  {kegg_f1s.mean():.3f} +/- {kegg_f1s.std():.3f}")
    print(f"{'KEGG-GA SVM':12s}  {kegg_ga_accs.mean():.3f} +/- {kegg_ga_accs.std():.3f}  {kegg_ga_f1s.mean():.3f} +/- {kegg_ga_f1s.std():.3f}")

    results = {
        "ga_svm": {
            "fold_accuracies": ga_accs.tolist(),
            "fold_f1s": ga_f1s.tolist(),
            "mean_accuracy": float(ga_accs.mean()),
            "std_accuracy": float(ga_accs.std()),
            "mean_f1": float(ga_f1s.mean()),
            "std_f1": float(ga_f1s.std()),
            "confusion_matrices": [m[2].tolist() for m in ga_metrics],
            "pareto_fronts": ga_fronts,
        },
        "kegg_svm": {
            "fold_accuracies": kegg_accs.tolist(),
            "fold_f1s": kegg_f1s.tolist(),
            "mean_accuracy": float(kegg_accs.mean()),
            "std_accuracy": float(kegg_accs.std()),
            "mean_f1": float(kegg_f1s.mean()),
            "std_f1": float(kegg_f1s.std()),
            "confusion_matrices": [m[2].tolist() for m in base_metrics],
        },
        "kegg_ga_svm": {
            "fold_accuracies": kegg_ga_accs.tolist(),
            "fold_f1s": kegg_ga_f1s.tolist(),
            "mean_accuracy": float(kegg_ga_accs.mean()),
            "std_accuracy": float(kegg_ga_accs.std()),
            "mean_f1": float(kegg_ga_f1s.mean()),
            "std_f1": float(kegg_ga_f1s.std()),
            "confusion_matrices": [m[2].tolist() for m in kegg_ga_metrics],
            "pareto_fronts": kegg_ga_fronts,
        },
    }
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {OUT_PATH}")
else:
    rf_ga_accs      = np.array([m[0] for m in rf_ga_metrics])
    rf_ga_f1s       = np.array([m[1] for m in rf_ga_metrics])
    rf_kegg_accs    = np.array([m[0] for m in rf_base_metrics])
    rf_kegg_f1s     = np.array([m[1] for m in rf_base_metrics])
    rf_kegg_ga_accs = np.array([m[0] for m in rf_kegg_ga_metrics])
    rf_kegg_ga_f1s  = np.array([m[1] for m in rf_kegg_ga_metrics])

    print("\n" + "=" * 65)
    print(f"{'':12s}  {'accuracy':>20s}  {'macro-F1':>20s}")
    print(f"{'GA RF':12s}  {rf_ga_accs.mean():.3f} +/- {rf_ga_accs.std():.3f}  {rf_ga_f1s.mean():.3f} +/- {rf_ga_f1s.std():.3f}")
    print(f"{'KEGG RF':12s}  {rf_kegg_accs.mean():.3f} +/- {rf_kegg_accs.std():.3f}  {rf_kegg_f1s.mean():.3f} +/- {rf_kegg_f1s.std():.3f}")
    print(f"{'KEGG-GA RF':12s}  {rf_kegg_ga_accs.mean():.3f} +/- {rf_kegg_ga_accs.std():.3f}  {rf_kegg_ga_f1s.mean():.3f} +/- {rf_kegg_ga_f1s.std():.3f}")

    rf_results = {
        "ga_rf": {
            "fold_accuracies": rf_ga_accs.tolist(),
            "fold_f1s": rf_ga_f1s.tolist(),
            "mean_accuracy": float(rf_ga_accs.mean()),
            "std_accuracy": float(rf_ga_accs.std()),
            "mean_f1": float(rf_ga_f1s.mean()),
            "std_f1": float(rf_ga_f1s.std()),
            "confusion_matrices": [m[2].tolist() for m in rf_ga_metrics],
            "pareto_fronts": rf_ga_fronts,
        },
        "kegg_rf": {
            "fold_accuracies": rf_kegg_accs.tolist(),
            "fold_f1s": rf_kegg_f1s.tolist(),
            "mean_accuracy": float(rf_kegg_accs.mean()),
            "std_accuracy": float(rf_kegg_accs.std()),
            "mean_f1": float(rf_kegg_f1s.mean()),
            "std_f1": float(rf_kegg_f1s.std()),
            "confusion_matrices": [m[2].tolist() for m in rf_base_metrics],
        },
        "kegg_ga_rf": {
            "fold_accuracies": rf_kegg_ga_accs.tolist(),
            "fold_f1s": rf_kegg_ga_f1s.tolist(),
            "mean_accuracy": float(rf_kegg_ga_accs.mean()),
            "std_accuracy": float(rf_kegg_ga_accs.std()),
            "mean_f1": float(rf_kegg_ga_f1s.mean()),
            "std_f1": float(rf_kegg_ga_f1s.std()),
            "confusion_matrices": [m[2].tolist() for m in rf_kegg_ga_metrics],
            "pareto_fronts": rf_kegg_ga_fronts,
        },
    }
    with open(OUT_PATH, "w") as f:
        json.dump(rf_results, f, indent=2)
    print(f"\nResults written to {OUT_PATH}")
    


