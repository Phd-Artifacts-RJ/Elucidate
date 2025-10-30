# shap_analysis.py
import shap
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.pipeline import Pipeline
import numpy as np # Added numpy

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.inspection import PartialDependenceDisplay
from sklearn.utils import check_random_state

# Set SHAP's visualize_feature to True to use matplotlib
shap.initjs()


# --- Explainer helpers (model-agnostic, pipeline-friendly) ---

@dataclass
class CFConstraints:
    immutable: List[str]
    lower_bounds: Dict[str, float]   # optional, can be {}
    upper_bounds: Dict[str, float]   # optional, can be {}

def _clip_to_bounds(x_df: pd.DataFrame, constraints: CFConstraints) -> pd.DataFrame:
    x = x_df.copy()
    for col, lb in constraints.lower_bounds.items():
        if col in x: x[col] = np.maximum(x[col], lb)
    for col, ub in constraints.upper_bounds.items():
        if col in x: x[col] = np.minimum(x[col], ub)
    return x

def _weighted_gower(a: np.ndarray, b: np.ndarray, weights: np.ndarray,
                    col_is_cat: np.ndarray) -> float:
    # Gower per-column distance in [0,1], then weighted average
    diffs = np.zeros_like(a, dtype=float)
    # numeric columns scaled to [0,1] by range in data sample (pass pre-scaled or unit-scale assumptions)
    diffs[~col_is_cat] = np.minimum(1.0, np.abs(a[~col_is_cat] - b[~col_is_cat]))
    # categorical mismatch is 0/1
    diffs[col_is_cat] = (a[col_is_cat] != b[col_is_cat]).astype(float)
    w = np.asarray(weights, dtype=float)
    w = w / (w.sum() + 1e-12)
    return float((w * diffs).sum())

def _candidate_perturbations(x0: pd.Series,
                             directions: Dict[str, int],
                             step_sizes: Dict[str, float],
                             constraints: CFConstraints) -> List[pd.Series]:
    cands = []
    for f, sgn in directions.items():
        if f in constraints.immutable or f not in x0.index: 
            continue
        step = step_sizes.get(f, 0.0) * sgn
        if step == 0.0: 
            continue
        x1 = x0.copy()
        x1[f] = x1[f] + step
        cands.append(x1)
    return cands

def find_counterfactual(model, x0: pd.DataFrame,  # 1-row DF
                        train_sample: pd.DataFrame, 
                        shap_abs_mean: pd.Series,   # abs mean SHAP by feature (aligned to x0 columns)
                        directions: Dict[str, int], # {feature: +1/-1/0} from your Stage-1
                        constraints: CFConstraints,
                        target_class: int = 1,
                        beam_width: int = 20,
                        max_steps: int = 30,
                        alpha: float = 1.0,  # distance weight
                        beta: float = 0.01,  # sparsity penalty
                        random_state: int = 42) -> Dict[str, Any]:
    rng = check_random_state(random_state)
    cols = x0.columns
    # crude categorical mask: bool if dtype is object or category
    col_is_cat = np.array([str(train_sample[c].dtype).startswith(("object", "category")) if c in train_sample else False for c in cols])

    # step sizes: 5% IQR per numeric; 1-hot/cat will be toggled via direction later if you encode pre-OHE
    step_sizes = {}
    for c in cols:
        if c in constraints.immutable: 
            continue
        if not col_is_cat[cols.get_loc(c)]:
            s = np.nanpercentile(train_sample[c].values, 75) - np.nanpercentile(train_sample[c].values, 25)
            step_sizes[c] = 0.05 * (s if s > 0 else (np.nanstd(train_sample[c].values) + 1e-6))

    # weights from SHAP (normalize)
    w = shap_abs_mean.reindex(cols).fillna(0.0).values
    w = w / (w.sum() + 1e-12)

    def score(x_series: pd.Series, parent_changes: int) -> Tuple[float, float]:
        x_df = _clip_to_bounds(pd.DataFrame([x_series.values], columns=cols), constraints)
        proba = model.predict_proba(x_df)[0, target_class]
        d = _weighted_gower(x0.values[0], x_df.values[0], w, col_is_cat)
        # minimize total objective; lower is better
        obj = alpha * d + beta * parent_changes
        return obj, proba

    # beam search
    beam = [(x0.iloc[0], 0)]  # (series, changes_count)
    best = None

    for _ in range(max_steps):
        expanded = []
        for state, k_changes in beam:
            obj, p = score(state, k_changes)
            pred = int(p >= 0.5)
            if pred != int(model.predict(x0)[0]):
                best = {"x_cf": pd.DataFrame([state.values], columns=cols),
                        "proba": p, "changes": k_changes, "objective": obj}
                break
            # expand neighbors
            for cand in _candidate_perturbations(state, directions, step_sizes, constraints):
                expanded.append((cand, k_changes + 1))
        if best is not None:
            break

        # select next beam by objective (use small noise to break ties)
        scored = []
        for cand, k in expanded:
            obj, p = score(cand, k)
            scored.append((obj + 1e-6 * rng.rand(), cand, k, p))
        scored.sort(key=lambda t: t[0])
        beam = [(cand, k) for (obj, cand, k, p) in scored[:beam_width]]

    return best if best is not None else {"x_cf": None, "proba": None, "changes": None, "objective": None}

def shap_top_interactions_for_tree(model, X_sample: pd.DataFrame, topk: int = 10):
    import shap
    try:
        expl = shap.TreeExplainer(model)
        inter = expl.shap_interaction_values(X_sample)
        # aggregate pairwise absolute interactions
        M = np.abs(inter).mean(axis=0)  # (features x features)
        idx = np.dstack(np.unravel_index(np.argsort(M.ravel())[::-1], M.shape))[0]
        pairs = []
        seen = set()
        for i, j in idx:
            if i == j: 
                continue
            key = tuple(sorted((i, j)))
            if key in seen: 
                continue
            seen.add(key)
            pairs.append((X_sample.columns[i], X_sample.columns[j], M[i, j]))
            if len(pairs) >= topk: break
        return pairs
    except Exception:
        return []

def plot_ice_pdp(ax, model, X: pd.DataFrame, feature: str, n_ice: int = 50):
    try:
        PartialDependenceDisplay.from_estimator(
            model, X, [feature], kind="both", centered=False,
            subsample=min(n_ice, len(X)), ax=ax
        )
    except Exception as e:
        ax.text(0.1, 0.5, f"PDP/ICE failed: {e}", transform=ax.transAxes)



def shap_rank_stability(model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                        n_bg_samples: int = 200, n_trials: int = 10,
                        random_state: int = 42) -> pd.DataFrame:
    rng = check_random_state(random_state)
    from shap import Explainer
    results = []
    for t in range(n_trials):
        # resample background
        bg = X_train.sample(min(n_bg_samples, len(X_train)), random_state=rng.randint(1e9))
        expl = _make_explainer(model, bg, X_train.columns.tolist())   # ✅ pipeline-safe
        sv = expl(X_test.sample(min(512, len(X_test)), random_state=rng.randint(1e9)))

        # works for both Explanation and raw arrays
        V = np.asarray(sv.values) if hasattr(sv, "values") else np.asarray(sv)
        abs_mean = np.abs(V).mean(axis=0)

        order = np.argsort(abs_mean)[::-1]
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)
        results.append(ranks)
    ranks_mat = np.vstack(results)
    features = X_test.columns
    return pd.DataFrame({
        "feature": features,
        "avg_rank": ranks_mat.mean(axis=0),
        "std_rank": ranks_mat.std(axis=0)
    }).sort_values("avg_rank")
    
def model_randomization_sanity(model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                               y_train: pd.Series, n_bg_samples: int = 200,
                               random_state: int = 42) -> float:
    """Return ratio: SHAP mass(original) / SHAP mass(randomized). Expect >> 1 if sane."""
    # Validate required inputs
    if X_train is None or X_test is None or y_train is None:
        raise ValueError(
            "model_randomization_sanity requires X_train, X_test and y_train. "
            f"Received types: X_train={type(X_train)}, X_test={type(X_test)}, y_train={type(y_train)}"
        )
    # Accept array-like y_train by converting to pandas.Series
    if not isinstance(y_train, pd.Series):
        try:
            y_train = pd.Series(y_train)
        except Exception:
            raise ValueError("y_train must be convertible to a pandas Series.")

 
 
    from shap import Explainer
    bg = X_train.sample(min(n_bg_samples, len(X_train)), random_state=random_state)
    expl = _make_explainer(model, bg, X_train.columns.tolist())   # ✅
    sv = expl(X_test)
    V1 = np.asarray(sv.values) if hasattr(sv, "values") else np.asarray(sv)
    mass_orig = np.abs(V1).mean()

    # randomized labels → retrain a copy (same class of estimator if possible)
    try:
        import sklearn.base as skbase
        m2 = skbase.clone(getattr(model, "steps")[-1][1].base_estimator if "calibratedclassifiercv" in str(model).lower() else model)
    except Exception:
        m2 = model
    y_rand = y_train.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    X_rand = X_train.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    m2.fit(X_rand, y_rand)
    expl2 = _make_explainer(m2, bg, X_train.columns.tolist())     # ✅
    sv2 = expl2(X_test)
    V2 = np.asarray(sv2.values) if hasattr(sv2, "values") else np.asarray(sv2)
    mass_rand = np.abs(V2).mean()

    return float(mass_orig / (mass_rand + 1e-12))


def _p1_wrapper(model, feature_names):
    """Returns f(X) -> P(class=1) for SHAP, accepting ndarray or DataFrame."""
    def predict_proba_p1(data):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data, columns=feature_names)
        return model.predict_proba(data)[:, 1]
    return predict_proba_p1

def _make_explainer(model, background_df, feature_names):
    """
    Prefer model-specific explainers via shap.Explainer; fall back to KernelExplainer
    on black-box pipelines. This avoids the brutal cost of Kernel when possible.
    """
    try:
        # shap.Explainer will dispatch to Tree/Linear/Deep when it recognizes the model
        return shap.Explainer(model, background_df, feature_names=feature_names)
    except Exception:
        # Robust fallback: model is a black-box pipeline → wrap predict_proba
        return shap.KernelExplainer(_p1_wrapper(model, feature_names), background_df)


def get_shap_values(model: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame,
                    max_bg: int = 200, max_explain: int = 1000):
    """
    GLOBAL SHAP for a sample of test points, plus global weights table.

    Returns:
        shap_values: np.ndarray or list-like (per SHAP API)
        explain_df:  pd.DataFrame used for explanation
        shap_global_df: DataFrame with columns [rank, feature, abs_mean_shap, mean_shap, std_shap]
    """
    feature_names = X_train.columns.tolist()

    # Background & explain sets (cap for tractability)
    background_df = shap.sample(X_train, min(max_bg, len(X_train)), random_state=42)
    explain_df    = shap.sample(X_test,  min(max_explain, len(X_test)), random_state=42)

    explainer = _make_explainer(model, background_df, feature_names)
    sv = explainer(explain_df)  # shap.Explanation (preferred) or values array (fallback)

    # Get matrix of SHAP values (n, d)
    if isinstance(sv, shap._explanation.Explanation):
        V = np.asarray(sv.values)
    else:
        V = np.asarray(sv)

    abs_mean = np.mean(np.abs(V), axis=0)
    mean_signed = np.mean(V, axis=0)
    std = np.std(V, axis=0)

    shap_global_df = (
        pd.DataFrame({
            "feature": feature_names,
            "abs_mean_shap": abs_mean,
            "mean_shap": mean_signed,
            "std_shap": std
        })
        .sort_values("abs_mean_shap", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )
    shap_global_df.insert(0, "rank", np.arange(1, len(shap_global_df) + 1))

    return sv, explain_df, shap_global_df

def compute_shap_direction(shap_global_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with columns [feature, sign] where sign in {-1,0,1}
    is the sign of the MEAN SHAP value for the positive class.
    """
    return pd.DataFrame({
        "feature": shap_global_df["feature"].values,
        "sign": np.sign(shap_global_df["mean_shap"].values).astype(int)
    })

def build_perturbation_subsets(shap_direction_df: pd.DataFrame,
                               numeric_feature_names: list[str],
                               immutable_columns: list[str] | None = None) -> dict:
    """
    Build Sup / Sdown / Simmutable as required by the paper's Stage-2 search.
    Convention: positive class = default. To reduce default risk:
      - If mean SHAP sign > 0, DECREASE the feature (→ Sdown)
      - If mean SHAP sign < 0, INCREASE the feature (→ Sup)
    Only numeric features are monotone-adjusted; categoricals are excluded.
    """
    immutable = set(immutable_columns or [])
    sign_map = dict(zip(shap_direction_df["feature"], shap_direction_df["sign"]))

    sup, sdown = [], []
    for f in numeric_feature_names:
        if f in immutable:
            continue
        s = sign_map.get(f, 0)
        if s < 0:
            sup.append(f)
        elif s > 0:
            sdown.append(f)

    return {
        "Sup": sorted(sup),
        "Sdown": sorted(sdown),
        "Simmutable": sorted(list(immutable)),
    }


# --- NEW FUNCTION FOR STEP 6 ---
def get_local_shap_explanation(model: Pipeline,
                               X_train: pd.DataFrame,
                               instance_df: pd.DataFrame,
                               max_bg: int = 200):
    """
    LOCAL SHAP for a SINGLE instance, returning shap.Explanation.
    """
    feature_names = X_train.columns.tolist()
    background_df = shap.sample(X_train, min(max_bg, len(X_train)), random_state=42)

    explainer = _make_explainer(model, background_df, feature_names)
    sv = explainer(instance_df)

    # Ensure a 1D explanation for the single row
    if isinstance(sv, shap._explanation.Explanation):
        values = np.asarray(sv.values).reshape(-1)
        base_value = np.asarray(sv.base_values).reshape(-1)[0]
    else:
        values = np.asarray(sv)[0]
        # Best-effort base value when using Kernel fallback
        base_value = explainer.expected_value if np.isscalar(explainer.expected_value) else explainer.expected_value[1]

    return shap.Explanation(
        values=values,
        base_values=base_value,
        data=instance_df.iloc[0],
        feature_names=feature_names
    )
