import sys
sys.path.insert(0, "src")

from svm import *
from rf import train_ga_rf, train_kegg_ga_rf, KEGG_RF_PARAM_GRID
from gene_selection import select_bc_features, genes_for_features
from feature_selection import select_knee_from_front
from pathlib import Path
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

def evaluate_knee(front, X_train, X_test, y_train, y_test,
                  model_kind, fold_num, model_label, metrics_list):
    """Pick the knee member of `front`, refit on train, score on test, save genes.

    Returns the knee dict (or None if kneed couldn't find one — in which case the
    best-CV member is fallback-stored under the knee key for downstream parity).
    """
    knee = select_knee_from_front(front)
    if knee is None:
        knee = max(front, key=lambda m: m["accuracy"])
    feats = np.array(knee["features"], dtype=int)
    if model_kind == "svm":
        clf = make_pipeline(StandardScaler(), LinearSVC(C=knee["C"], max_iter=5000, dual="auto"))
    else:
        clf = make_pipeline(RandomForestClassifier(**knee["cfg"], random_state=42, n_jobs=-1))
    clf.fit(X_train[:, feats], y_train)
    yhat = clf.predict(X_test[:, feats])
    acc = accuracy_score(y_test, yhat)
    f1 = f1_score(y_test, yhat, average="macro")
    cm = confusion_matrix(y_test, yhat)
    metrics_list.append((acc, f1, cm))
    with open(f"data/{model_label}_knee_genes_fold_{fold_num}.txt", "w") as gf:
        gf.write("\n".join(genes_for_features(feats)) + "\n")
    hp = knee.get("C", knee.get("cfg"))
    print(f"  └─ knee — acc={acc:.3f}  macro-F1={f1:.3f}  features={len(feats)}  hp={hp}")
    return knee


def save_pareto_fronts_stacked(front, out_path):
    """Write every Pareto-front member's gene list to one stacked text file.

    Format per member:
        >>> member_<NNN> | acc=... | n_features=... | C=... [| n_bc=...]
        GENE1
        GENE2
        ...
        <blank line>

    Headers are grep-able on `^>>>`; gene blocks paste directly into NCBI GO.
    """
    with open(out_path, "w") as f:
        f.write(f"# Pareto front — {len(front)} members\n")
        f.write("# Blocks start with '>>>' header; gene symbols follow one per line.\n\n")
        for i, m in enumerate(front):
            genes = genes_for_features(m["features"])
            header = [f"member_{i:03d}",
                      f"acc={m['accuracy']:.4f}",
                      f"n_features={m['n_features']}"]
            if "C" in m:
                header.append(f"C={m['C']}")
            if "cfg" in m:
                header.append(f"cfg={m['cfg']}")
            if m.get("n_bc") is not None:
                header.append(f"n_bc={m['n_bc']}")
            f.write(">>> " + " | ".join(header) + "\n")
            for g in genes:
                f.write(g + "\n")
            f.write("\n")


def save_pareto_member_genes(front, model_label, fold_num):
    """Write one gene-symbol file per Pareto-front member.

    Layout: data/pareto_genes/<model_label>/fold_<N>_member_<MMM>.txt
    """
    out = Path("data/pareto_genes") / model_label
    out.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(front):
        symbols = genes_for_features(m["features"])
        path = out / f"fold_{fold_num}_member_{i:03d}.txt"
        with open(path, "w") as f:
            f.write("\n".join(symbols) + "\n")


ga_metrics = []
base_metrics = []
kegg_ga_metrics = []
ga_fronts = []
kegg_ga_fronts = []
ga_knee_metrics = []
kegg_ga_knee_metrics = []

rf_ga_metrics = []
rf_base_metrics = []
rf_kegg_ga_metrics = []
rf_ga_fronts = []
rf_kegg_ga_fronts = []
rf_ga_knee_metrics = []
rf_kegg_ga_knee_metrics = []

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
            save_pareto_fronts_stacked(front, f"data/svm_ga_genes_fold_{fold_num}.txt")
            print(f"Fold {fold_num} GA      — acc={acc:.3f}  macro-F1={f1:.3f}  features={len(selected)}  C={best_C}  pareto={len(front)}")
            evaluate_knee(front, X_train, X_test, y_train, y_test,
                          "svm", fold_num, "svm_ga", ga_knee_metrics)

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
            evaluate_knee(front, X_train, X_test, y_train, y_test,
                          "svm", fold_num, "svm_kegg_ga", kegg_ga_knee_metrics)

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
            save_pareto_member_genes(front, "rf_ga", fold_num)
            print(f"Fold {fold_num} GA RF      — acc={acc:.3f}  macro-F1={f1:.3f}  features={len(selected)}  pareto={len(front)}")
            evaluate_knee(front, X_train, X_test, y_train, y_test,
                          "rf", fold_num, "rf_ga", rf_ga_knee_metrics)

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
            save_pareto_member_genes(front, "rf_kegg_ga", fold_num)
            print(f"Fold {fold_num} KEGG-GA RF — acc={acc:.3f}  macro-F1={f1:.3f}  features={len(selected)}  pareto={len(front)}")
            evaluate_knee(front, X_train, X_test, y_train, y_test,
                          "rf", fold_num, "rf_kegg_ga", rf_kegg_ga_knee_metrics)


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

    ga_knee_accs      = np.array([m[0] for m in ga_knee_metrics])
    ga_knee_f1s       = np.array([m[1] for m in ga_knee_metrics])
    kegg_ga_knee_accs = np.array([m[0] for m in kegg_ga_knee_metrics])
    kegg_ga_knee_f1s  = np.array([m[1] for m in kegg_ga_knee_metrics])

    print("\n" + "=" * 65)
    print(f"{'':20s}  {'accuracy':>20s}  {'macro-F1':>20s}")
    print(f"{'GA SVM (best)':20s}  {ga_accs.mean():.3f} +/- {ga_accs.std():.3f}  {ga_f1s.mean():.3f} +/- {ga_f1s.std():.3f}")
    print(f"{'GA SVM (knee)':20s}  {ga_knee_accs.mean():.3f} +/- {ga_knee_accs.std():.3f}  {ga_knee_f1s.mean():.3f} +/- {ga_knee_f1s.std():.3f}")
    print(f"{'KEGG SVM':20s}  {kegg_accs.mean():.3f} +/- {kegg_accs.std():.3f}  {kegg_f1s.mean():.3f} +/- {kegg_f1s.std():.3f}")
    print(f"{'KEGG-GA SVM (best)':20s}  {kegg_ga_accs.mean():.3f} +/- {kegg_ga_accs.std():.3f}  {kegg_ga_f1s.mean():.3f} +/- {kegg_ga_f1s.std():.3f}")
    print(f"{'KEGG-GA SVM (knee)':20s}  {kegg_ga_knee_accs.mean():.3f} +/- {kegg_ga_knee_accs.std():.3f}  {kegg_ga_knee_f1s.mean():.3f} +/- {kegg_ga_knee_f1s.std():.3f}")

    def _stat_block(metrics):
        accs = np.array([m[0] for m in metrics])
        f1s = np.array([m[1] for m in metrics])
        return {
            "fold_accuracies": accs.tolist(),
            "fold_f1s": f1s.tolist(),
            "mean_accuracy": float(accs.mean()),
            "std_accuracy": float(accs.std()),
            "mean_f1": float(f1s.mean()),
            "std_f1": float(f1s.std()),
            "confusion_matrices": [m[2].tolist() for m in metrics],
        }

    results = {
        "ga_svm":           {**_stat_block(ga_metrics),         "pareto_fronts": ga_fronts},
        "ga_svm_knee":      _stat_block(ga_knee_metrics),
        "kegg_svm":         _stat_block(base_metrics),
        "kegg_ga_svm":      {**_stat_block(kegg_ga_metrics),    "pareto_fronts": kegg_ga_fronts},
        "kegg_ga_svm_knee": _stat_block(kegg_ga_knee_metrics),
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
    rf_ga_knee_accs      = np.array([m[0] for m in rf_ga_knee_metrics])
    rf_ga_knee_f1s       = np.array([m[1] for m in rf_ga_knee_metrics])
    rf_kegg_ga_knee_accs = np.array([m[0] for m in rf_kegg_ga_knee_metrics])
    rf_kegg_ga_knee_f1s  = np.array([m[1] for m in rf_kegg_ga_knee_metrics])

    print("\n" + "=" * 65)
    print(f"{'':20s}  {'accuracy':>20s}  {'macro-F1':>20s}")
    print(f"{'GA RF (best)':20s}  {rf_ga_accs.mean():.3f} +/- {rf_ga_accs.std():.3f}  {rf_ga_f1s.mean():.3f} +/- {rf_ga_f1s.std():.3f}")
    print(f"{'GA RF (knee)':20s}  {rf_ga_knee_accs.mean():.3f} +/- {rf_ga_knee_accs.std():.3f}  {rf_ga_knee_f1s.mean():.3f} +/- {rf_ga_knee_f1s.std():.3f}")
    print(f"{'KEGG RF':20s}  {rf_kegg_accs.mean():.3f} +/- {rf_kegg_accs.std():.3f}  {rf_kegg_f1s.mean():.3f} +/- {rf_kegg_f1s.std():.3f}")
    print(f"{'KEGG-GA RF (best)':20s}  {rf_kegg_ga_accs.mean():.3f} +/- {rf_kegg_ga_accs.std():.3f}  {rf_kegg_ga_f1s.mean():.3f} +/- {rf_kegg_ga_f1s.std():.3f}")
    print(f"{'KEGG-GA RF (knee)':20s}  {rf_kegg_ga_knee_accs.mean():.3f} +/- {rf_kegg_ga_knee_accs.std():.3f}  {rf_kegg_ga_knee_f1s.mean():.3f} +/- {rf_kegg_ga_knee_f1s.std():.3f}")

    def _stat_block(metrics):
        accs = np.array([m[0] for m in metrics])
        f1s = np.array([m[1] for m in metrics])
        return {
            "fold_accuracies": accs.tolist(),
            "fold_f1s": f1s.tolist(),
            "mean_accuracy": float(accs.mean()),
            "std_accuracy": float(accs.std()),
            "mean_f1": float(f1s.mean()),
            "std_f1": float(f1s.std()),
            "confusion_matrices": [m[2].tolist() for m in metrics],
        }

    rf_results = {
        "ga_rf":           {**_stat_block(rf_ga_metrics),         "pareto_fronts": rf_ga_fronts},
        "ga_rf_knee":      _stat_block(rf_ga_knee_metrics),
        "kegg_rf":         _stat_block(rf_base_metrics),
        "kegg_ga_rf":      {**_stat_block(rf_kegg_ga_metrics),    "pareto_fronts": rf_kegg_ga_fronts},
        "kegg_ga_rf_knee": _stat_block(rf_kegg_ga_knee_metrics),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(rf_results, f, indent=2)
    print(f"\nResults written to {OUT_PATH}")
    


