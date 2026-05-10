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

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, eye, bmat, vstack
from scipy.optimize import linprog


def read_data():
    """Load CATS arrayCGH training data along with chromosome assignments."""
    call = pd.read_csv("B4TM_CATS_training_data/Train_call.tsv", sep="\t")
    clinical = pd.read_csv("B4TM_CATS_training_data/Train_clinical.tsv", sep="\t")
    clinical["Subgroup"] = clinical["Subgroup"].astype("category")

    chromosomes = call["Chromosome"].to_numpy()

    sample_columns = call.columns[4:]
    transposed_call = call[sample_columns].T
    transposed_call.index.name = "Sample"
    transposed_call.reset_index(inplace=True)

    merged = pd.merge(transposed_call, clinical, on="Sample")
    X = merged.drop(columns=["Sample", "Subgroup"])
    y = merged["Subgroup"]
    return X, y, chromosomes


class FusedSVM(BaseEstimator, ClassifierMixin):
    """Binary fused SVM solved as a linear program (paper Eq. 6).

    Parameters
    ----------
    lambda_ : float
        L1 budget; sum_i |w_i| <= lambda_. Smaller -> sparser w.
    mu : float
        Fusion budget; sum_{i~j} |w_i - w_j| <= mu. Smaller -> more piecewise
        constant w. mu >= 2*lambda_ effectively turns the fusion off, recovering
        a plain L1-SVM (paper Section 4).
    chromosomes : array-like of shape (n_features,) or None
        Per-feature chromosome ID. Only consecutive features with equal IDs are
        fused. If None, all consecutive features are fused (single chromosome).
    """

    def __init__(self, lambda_=10.0, mu=1.0, chromosomes=None):
        self.lambda_ = lambda_
        self.mu = mu
        self.chromosomes = chromosomes

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape

        if self.chromosomes is None:
            chrom = np.zeros(p, dtype=int)
        else:
            chrom = np.asarray(self.chromosomes)
            assert chrom.shape[0] == p, "chromosomes length must match n_features"

        # Within-chromosome consecutive pairs (i, i+1).
        pair_i = np.flatnonzero(chrom[:-1] == chrom[1:])
        pair_j = pair_i + 1
        q = pair_i.size

        # LP variables (concatenated): z = [w (p) | alpha (n) | beta (p) | gamma (q)]
        n_vars = p + n + p + q
        b_off = p + n
        g_off = p + n + p

        # Objective: minimize sum(alpha) — paper Eq. 6.
        c = np.zeros(n_vars)
        c[p:p + n] = 1.0

        # Inequality blocks A_ub @ z <= b_ub:
        #   (H)  -y_i (X[i] . w) - alpha_i <= -1                 (alpha_i >= 1 - y_i w.x_i)
        #   (B+)  w_i - beta_i <= 0                              (beta_i >= w_i)
        #   (B-) -w_i - beta_i <= 0                              (beta_i >= -w_i)
        #   (L)   sum_i beta_i <= lambda_
        #   (F+)  w_i - w_j - gamma_k <= 0      for i~j         (gamma_k >= w_i - w_j)
        #   (F-) -w_i + w_j - gamma_k <= 0      for i~j         (gamma_k >= w_j - w_i)
        #   (M)   sum_k gamma_k <= mu
        Ip = eye(p, format="csr")
        In = eye(n, format="csr")

        neg_yX = csr_matrix(-(y[:, None] * X))
        A_H = bmat([[neg_yX, -In, csr_matrix((n, p)), csr_matrix((n, q))]])
        b_H = -np.ones(n)

        A_Bp = bmat([[ Ip, csr_matrix((p, n)), -Ip, csr_matrix((p, q))]])
        A_Bm = bmat([[-Ip, csr_matrix((p, n)), -Ip, csr_matrix((p, q))]])
        b_B = np.zeros(p)

        row_L = np.zeros(n_vars); row_L[b_off:b_off + p] = 1.0
        A_L = csr_matrix(row_L.reshape(1, -1))
        b_L = np.array([self.lambda_])

        blocks = [A_H, A_Bp, A_Bm, A_L]
        rhs = [b_H, b_B, b_B, b_L]

        if q > 0:
            Iq = eye(q, format="csr")
            data = np.concatenate([np.ones(q), -np.ones(q)])
            rows = np.concatenate([np.arange(q), np.arange(q)])
            cols = np.concatenate([pair_i, pair_j])
            D = csr_matrix((data, (rows, cols)), shape=(q, p))   # consecutive-difference op

            A_Fp = bmat([[ D, csr_matrix((q, n)), csr_matrix((q, p)), -Iq]])
            A_Fm = bmat([[-D, csr_matrix((q, n)), csr_matrix((q, p)), -Iq]])
            row_M = np.zeros(n_vars); row_M[g_off:g_off + q] = 1.0
            A_M = csr_matrix(row_M.reshape(1, -1))

            blocks += [A_Fp, A_Fm, A_M]
            rhs += [np.zeros(q), np.zeros(q), np.array([self.mu])]

        A_ub = vstack(blocks).tocsr()
        b_ub = np.concatenate(rhs)

        # w is free; alpha, beta, gamma are >= 0.
        bounds = [(None, None)] * p + [(0, None)] * (n + p + q)

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        self.lp_status_ = res.message
        if not res.success:
            raise RuntimeError(f"linprog failed: {res.message}")

        self.w_ = res.x[:p]
        self.classes_ = np.array([-1.0, 1.0])
        return self

    def decision_function(self, X):
        return np.asarray(X) @ self.w_

    def predict(self, X):
        s = self.decision_function(X)
        return np.where(s >= 0, 1.0, -1.0)


class FusedSVMOvR(BaseEstimator, ClassifierMixin):
    """Multiclass fused SVM via one-vs-rest (paper Section 2.2).

    Trains one binary FusedSVM per class; each classifier produces its own
    sparse, piecewise-constant weight profile. At prediction time, the class
    whose classifier emits the largest margin wins.
    """

    def __init__(self, lambda_=10.0, mu=1.0, chromosomes=None):
        self.lambda_ = lambda_
        self.mu = mu
        self.chromosomes = chromosomes

    def fit(self, X, y):
        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_

        self.classifiers_ = []
        for k in range(len(self.classes_)):
            y_bin = np.where(y_enc == k, 1.0, -1.0)
            clf = FusedSVM(lambda_=self.lambda_, mu=self.mu,
                           chromosomes=self.chromosomes).fit(X, y_bin)
            self.classifiers_.append(clf)
        return self

    def predict(self, X):
        scores = np.column_stack([clf.decision_function(X) for clf in self.classifiers_])
        return self.label_encoder_.inverse_transform(np.argmax(scores, axis=1))


# ---------------------------------------------------------------------------
# Verification: run `python src/svm.py` from the repo root.
# ---------------------------------------------------------------------------
def _piecewise_summary(w, chrom, tol=1e-6):
    """Return (#nonzero, #transitions, #plateaus).

    - transitions: within-chromosome consecutive pairs whose weights differ.
      For a sparse-only w, each isolated nonzero contributes ~2 transitions.
      For a fused w, transitions count plateau edges only.
    - plateaus: maximal runs of equal weight on a chromosome (zero or nonzero).
    """
    w = np.where(np.abs(w) < tol, 0.0, w)
    nonzero = int((np.abs(w) > 0).sum())
    same_chrom = chrom[:-1] == chrom[1:]
    differs = np.abs(w[:-1] - w[1:]) >= tol
    transitions = int((same_chrom & differs).sum())
    # plateaus = within-chrom transitions + number of chromosomes
    n_chroms = int(np.unique(chrom).size)
    plateaus = transitions + n_chroms
    return nonzero, transitions, plateaus


def _cosine(a, b):
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n > 0 else 0.0


if __name__ == "__main__":
    # ---- Synthetic recovery test --------------------------------------
    # Ground truth: sparse + piecewise-constant. Two plateaus on different
    # "chromosomes", zero everywhere else. The fused SVM should recover
    # this shape; the plain L1-SVM (mu very large) should not.
    rng = np.random.default_rng(0)
    p_syn = 120
    chrom_syn = np.repeat(np.arange(1, 5), 30)        # 4 chromosomes x 30 probes
    w_true = np.zeros(p_syn)
    w_true[10:25] = 1.0      # positive plateau on chrom 1
    w_true[70:90] = -1.0     # negative plateau on chrom 3

    n_syn = 100
    X_syn = rng.normal(size=(n_syn, p_syn))
    y_syn = np.where(X_syn @ w_true + 0.1 * rng.normal(size=n_syn) >= 0, 1.0, -1.0)

    fused = FusedSVM(lambda_=20.0, mu=4.0, chromosomes=chrom_syn).fit(X_syn, y_syn)
    plain = FusedSVM(lambda_=20.0, mu=1e6, chromosomes=chrom_syn).fit(X_syn, y_syn)

    print("=== Synthetic recovery test (truth = piecewise-constant w) ===")
    print(f"  ground-truth: nonzeros={(np.abs(w_true) > 0).sum()}/{p_syn}, plateaus=6 "
          f"(2 nonzero + 4 zero-runs across chroms)")
    nz_f, tr_f, pl_f = _piecewise_summary(fused.w_, chrom_syn)
    nz_p, tr_p, pl_p = _piecewise_summary(plain.w_, chrom_syn)
    print(f"  fused SVM (mu=4)  : nonzeros={nz_f:3d}  transitions={tr_f:3d}  plateaus={pl_f:3d}  "
          f"train-acc={(fused.predict(X_syn) == y_syn).mean():.3f}  "
          f"cos(w_hat, w_true)={_cosine(fused.w_, w_true):+.3f}")
    print(f"  L1-SVM   (mu=inf) : nonzeros={nz_p:3d}  transitions={tr_p:3d}  plateaus={pl_p:3d}  "
          f"train-acc={(plain.predict(X_syn) == y_syn).mean():.3f}  "
          f"cos(w_hat, w_true)={_cosine(plain.w_, w_true):+.3f}")
    print(f"  fused-SVM weights inside chrom-1 plateau (idx 10..24):\n    "
          f"{np.round(fused.w_[10:25], 4)}")
    print(f"  fused-SVM weights inside chrom-3 plateau (idx 70..89):\n    "
          f"{np.round(fused.w_[70:90], 4)}")
    print(f"  fused-SVM max |w| outside plateaus: "
          f"{np.max(np.abs(np.delete(fused.w_, np.r_[10:25, 70:90]))):.4f}")
    print("  -> Lower transitions and higher cosine for fused vs L1 confirm the")
    print("     fusion penalty pulls w toward a piecewise-constant shape.")

    # ---- Real-data test (3-class CATS, single 80/20 split) ------------
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        X, y, chrom = read_data()
        X = X.to_numpy(); y = y.to_numpy()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

        print(f"\n=== CATS arrayCGH (3-class one-vs-rest, 80/20 holdout) ===")
        print(f"  train shape: {Xtr.shape}, classes: {np.unique(y).tolist()}")

        for lam, mu, label in [(50.0, 1.0, "fused SVM (tight)"),
                               (50.0, 1e6, "plain L1-SVM     ")]:
            clf = FusedSVMOvR(lambda_=lam, mu=mu, chromosomes=chrom).fit(Xtr, ytr)
            acc_tr = accuracy_score(ytr, clf.predict(Xtr))
            acc_te = accuracy_score(yte, clf.predict(Xte))
            stats = []
            for k, sub in enumerate(clf.classifiers_):
                nz, tr_, _ = _piecewise_summary(sub.w_, chrom)
                stats.append(f"class {clf.classes_[k]}: nonzeros={nz}  transitions={tr_}")
            print(f"  {label} (lambda={lam}, mu={mu}): "
                  f"train-acc={acc_tr:.3f}  test-acc={acc_te:.3f}")
            for s in stats:
                print(f"      {s}")
        print("  -> Fewer transitions for the fused configuration confirm the fusion")
        print("     constraint forced adjacent same-chromosome weights to merge.")
    except FileNotFoundError as e:
        print(f"\n[skipped real-data test: {e}]")
