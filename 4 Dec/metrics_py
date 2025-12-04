# metrics.py
import numpy as np
# --- MODIFIED: Added f1_score and recall_score ---
from sklearn.metrics import (
    roc_auc_score, 
    brier_score_loss, 
    roc_curve, 
    accuracy_score,
    f1_score,
    recall_score
)
from scipy.stats import beta

# ---------- helpers ----------
def prevalence_threshold(y_train):
    """τ = positive rate in TRAIN fold; used as PCC threshold."""
    y_train = np.asarray(y_train).astype(int)
    return float((y_train == 1).mean())

def ks_statistic(y_true, y_score):
    order = np.argsort(y_score)
    y = np.asarray(y_true)[order]
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    cpos = np.cumsum(y == 1) / max(pos, 1)
    cneg = np.cumsum(y == 0) / max(neg, 1)
    return float(np.max(np.abs(cpos - cneg)))

def gini_from_auc(auc):
    return 2.0 * auc - 1.0

def partial_gini(y_true, y_score, b=0.4):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    mask = y_score <= b
    if mask.sum() < 2 or len(np.unique(y_true[mask])) < 2:
        return np.nan
    auc = roc_auc_score(y_true[mask], y_score[mask])
    return float(gini_from_auc(auc))

# ---------- Hand’s H-measure (pure Python) ----------
def h_measure(y_true, y_score, severity=(2.0, 2.0), pi_plus=None, n_grid=1000):
    """
    Normalized H (Hand 2009) as used in Lessmann et al.:
    - expected MIN loss under Beta(a,b) cost distribution
    - normalized so random = 0, perfect = 1
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    if y_true.ndim != 1 or y_score.ndim != 1 or y_true.size != y_score.size:
        raise ValueError("h_measure: y_true and y_score must be 1D arrays of equal length.")
    if np.allclose(y_score.min(), y_score.max()):
        return np.nan

    if pi_plus is None:
        pi_plus = float((y_true == 1).mean())
    pi_minus = 1.0 - pi_plus

    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr

    a, b = severity
    c = beta.ppf(np.linspace(1e-6, 1 - 1e-6, n_grid), a, b)
    w = beta.pdf(c, a, b)
    w /= w.sum()

    losses_min = []
    for ci in c:
        # L_c(t) = c*pi+*FNR + (1-c)*pi-*FPR; take min over thresholds (scan ROC)
        L = ci * pi_plus * fnr + (1.0 - ci) * pi_minus * fpr
        losses_min.append(L.min())
    EMC = float(np.dot(w, np.array(losses_min)))

    baseline_min = np.minimum(c * pi_plus, (1.0 - c) * pi_minus)
    EMC0 = float(np.dot(w, baseline_min))
    if EMC0 <= 0:
        return np.nan

    H = 1.0 - EMC / EMC0
    return float(np.clip(H, 0.0, 1.0))

# ---------- bundle: compute all metrics for one fold ----------
def compute_metrics(y_tr, y_te, p_te, pg_cut=0.4, h_severity=(2.0, 2.0)):
    """
    Returns dict with AUC, PCC, F1, Recall, BS, KS, PG, H for a single test fold.
    PCC, F1, and Recall use τ = prevalence(y_tr).
    """
    tau = prevalence_threshold(y_tr)
    # y_pred is used for PCC, F1, and Recall
    y_pred = (p_te > tau).astype(int) 

    auc = roc_auc_score(y_te, p_te)
    pcc = accuracy_score(y_te, y_pred)
    # --- NEW: Calculate F1 and Recall ---
    f1 = f1_score(y_te, y_pred, zero_division=0)
    recall = recall_score(y_te, y_pred, zero_division=0)
    # --- END NEW ---
    bs  = brier_score_loss(y_te, p_te)
    ks  = ks_statistic(y_te, p_te)
    pg  = partial_gini(y_te, p_te, b=pg_cut)
    H   = h_measure(y_te, p_te, severity=h_severity, pi_plus=float((y_tr == 1).mean()))

    # --- MODIFIED: Added F1 and Recall to the returned dictionary ---
    return {
        "AUC": auc, 
        "PCC": pcc, 
        "F1": f1,
        "Recall": recall,
        "BS": bs, 
        "KS": ks, 
        "PG": pg, 
        "H": H
    }
