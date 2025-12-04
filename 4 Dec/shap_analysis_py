# shap_analysis.py
import hashlib
import shap
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.pipeline import Pipeline
import numpy as np  # Added numpy
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.inspection import PartialDependenceDisplay
from sklearn.utils import check_random_state
from sklearn.model_selection import StratifiedShuffleSplit

# Set SHAP's visualize_feature to True to use matplotlib
shap.initjs()


# --- Stratified sampling helper ---

def _stratified_sample(X: pd.DataFrame, y: pd.Series, n_samples: int, random_state: int):
    """
    Return a stratified sample of X (and y) with at most n_samples rows.
    If n_samples >= len(X), returns X, y unchanged.
    """
    n_samples = min(n_samples, len(X))
    if n_samples == len(X):
        return X.copy(), y.copy()

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=n_samples, random_state=random_state
    )
    idx_train, idx_sample = next(splitter.split(X, y))
    return X.iloc[idx_sample].copy(), y.iloc[idx_sample].copy()


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
    """
    Compute SHAP rank stability across multiple resamples.
    Uses the inferred feature names from the preprocessing pipeline to keep lengths aligned.
    """
    rng = check_random_state(random_state)
    from shap import Explainer
    preprocess, _ = _split_pipeline(model)
    results = []
    feature_names = None

    for _ in range(n_trials):
        bg = X_train.sample(min(n_bg_samples, len(X_train)), random_state=rng.randint(1e9))
        expl = _make_explainer(model, bg, X_train.columns.tolist())
        X_test_sample = X_test.sample(min(512, len(X_test)), random_state=rng.randint(1e9))
        sv = expl(_transform_data_for_explainer(model, X_test_sample))

        V = np.asarray(sv.values) if hasattr(sv, "values") else np.asarray(sv)
        abs_mean = np.abs(V).mean(axis=0)

        feature_names = _infer_feature_names(preprocess, X_test.columns, V.shape[1])

        order = np.argsort(abs_mean)[::-1]
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)
        results.append(ranks)

    ranks_mat = np.vstack(results)
    n = min(ranks_mat.shape[1], len(feature_names))
    ranks_mat = ranks_mat[:, :n]
    features = feature_names[:n]
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
    sv = expl(_transform_data_for_explainer(model, X_test))
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
    sv2 = expl2(_transform_data_for_explainer(m2, X_test))
    V2 = np.asarray(sv2.values) if hasattr(sv2, "values") else np.asarray(sv2)
    mass_rand = np.abs(V2).mean()

    return float(mass_orig / (mass_rand + 1e-12))



def _p1_wrapper(model, feature_names):
    """
    Returns f(X) -> P(class=1) for SHAP, accepting ndarray or DataFrame.
    Handles both raw-feature inputs and already-transformed numeric matrices.
    """
    preprocess, estimator = _split_pipeline(model)
    raw_dim = len(feature_names)

    def predict_proba_p1(data):
        # Common happy path: DataFrame with the expected raw columns
        if isinstance(data, pd.DataFrame):
            try:
                return model.predict_proba(data[feature_names])[:, 1]
            except Exception:
                try:
                    return model.predict_proba(data)[:, 1]
                except Exception:
                    pass  # fall through to numeric handling

        arr = _to_numeric_matrix(data)
        n_features = arr.shape[1] if hasattr(arr, "shape") else None

        # If the incoming matrix matches the raw feature count, align columns
        if n_features == raw_dim:
            try:
                df = pd.DataFrame(arr, columns=feature_names)
                return model.predict_proba(df)[:, 1]
            except Exception:
                pass

        # If it looks already transformed and we have access to the estimator,
        # bypass the preprocessing step.
        if estimator is not None and preprocess is not None and n_features is not None:
            try:
                return estimator.predict_proba(arr)[:, 1]
            except Exception:
                pass

        # Last resort: let the pipeline figure it out
        return model.predict_proba(arr)[:, 1]

    return predict_proba_p1


def _split_pipeline(model):
    """Return (preprocess, estimator) whether model is a Pipeline or bare estimator."""
    if isinstance(model, Pipeline):
        preprocess = model[:-1]
        estimator = model[-1]
    else:
        preprocess = None
        estimator = model
    return preprocess, estimator


def _to_numeric_matrix(data):
    """Best-effort conversion to a dense numeric matrix for SHAP."""
    if hasattr(data, "toarray"):
        data = data.toarray()
    try:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        df = df.apply(pd.to_numeric, errors="coerce")
        return df.to_numpy(dtype=np.float64, copy=False)
    except Exception:
        arr = np.asarray(data)
        if arr.dtype.kind == "O":
            try:
                arr = arr.astype(np.float64)
            except Exception:
                pass
        return arr


def _transform_data_for_explainer(model, data):
    """
    Robust version for SHAP:
    - Always returns a numeric DataFrame (float64)
    - Converts object/categorical columns to factorized integer codes
    - Preserves column order
    - Handles NaN safely
    This prevents SHAP 'isfinite' and dtype errors.
    """
    # Convert array-like to DataFrame first
    if not isinstance(data, pd.DataFrame):
        try:
            data = pd.DataFrame(data)
        except Exception:
            data = pd.DataFrame(np.asarray(data))

    df = data.copy()

    # 1. Convert ALL object/categorical columns to numeric codes
    for col in df.columns:
        if df[col].dtype == "object" or str(df[col].dtype).startswith("category"):
            try:
                # Convert to string first, then factorize
                codes, _ = pd.factorize(df[col].astype(str))
                df[col] = codes
            except Exception:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2. Coerce all columns to numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # 3. Ensure float64 matrix (what SHAP expects)
    return df.astype(np.float64)



def _make_explainer(model, background_df, feature_names):
    """
    Build a SHAP explainer that is robust to sklearn Pipelines and mixed dtypes.

    Strategy:
    - For generic models (including logistic regression in a pipeline), we use
      shap.Explainer on a wrapper f(X) -> P(class=1) that calls the full pipeline.
    - For plain tree models *without* a preprocessing pipeline, we still try
      TreeExplainer on a numeric matrix for speed.
    """
    preprocess, estimator = _split_pipeline(model)
    class_name = estimator.__class__.__name__.lower()

    # Keep a DataFrame-like view for the generic wrapper explainer
    X_bg_raw = _transform_data_for_explainer(model, background_df)
    feature_names_arg = feature_names

    # Try a fast path ONLY for bare tree models (no preprocessing)
    try:
        if preprocess is None and any(
            k in class_name for k in ["forest", "boost", "xgb", "gbm", "gradientboost"]
        ):
            # Ensure numeric matrix for the tree estimator
            X_bg_num = _to_numeric_matrix(X_bg_raw)
            return shap.TreeExplainer(estimator, X_bg_num, feature_names=feature_names_arg)

        # For everything else (including logistic regression and other linear models),
        # use a generic pipeline-aware wrapper. This avoids feeding raw strings into
        # LinearExplainer and prevents the 'isfinite' dtype error.
        f_p1 = _p1_wrapper(model, feature_names)
        return shap.Explainer(f_p1, X_bg_raw, feature_names=feature_names_arg)

    except Exception:
        # Ultimate fallback: KernelExplainer on the same wrapper
        f_p1 = _p1_wrapper(model, feature_names)
        return shap.KernelExplainer(f_p1, X_bg_raw)


def _infer_feature_names(preprocess, raw_feature_names, n_features):
    """
    Best-effort retrieval of feature names after preprocessing to align SHAP values.
    """
    if preprocess is not None:
        for getter in (
            lambda: preprocess.get_feature_names_out(raw_feature_names),
            lambda: preprocess.get_feature_names_out(),
        ):
            try:
                names = list(getter())
                if len(names) == n_features:
                    return names
            except Exception:
                pass
        try:
            first = preprocess.steps[0][1] if hasattr(preprocess, "steps") and preprocess.steps else None
            if first is not None and hasattr(first, "get_feature_names_out"):
                names = list(first.get_feature_names_out(raw_feature_names))
                if len(names) == n_features:
                    return names
        except Exception:
            pass

    if len(raw_feature_names) == n_features:
        return list(raw_feature_names)

    return [f"feature_{i}" for i in range(n_features)]


def _coerce_df_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a DataFrame to numeric columns for SHAP plotting.
    Non-numeric columns are factorized to integer codes to avoid isfinite errors.
    """
    out = df.copy()
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            try:
                codes, _ = pd.factorize(out[col].astype(str))
                out[col] = codes
            except Exception:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out

def _hash_model_and_bg(model, background_df: pd.DataFrame) -> str:
    """
    Crude but stable hash of the model plus background shape/content.
    Used to cache SHAP explainers across Streamlit reruns.
    """
    h = hashlib.sha1()
    h.update(str(model).encode("utf-8"))
    h.update(str(background_df.shape).encode("utf-8"))
    try:
        sample = background_df.head(5).to_csv(index=False).encode("utf-8")
        h.update(sample)
    except Exception:
        pass
    return h.hexdigest()


@st.cache_resource
def _get_explainer_cached(_model, background_df, feature_names, cache_key: str):
    """
    Cache SHAP explainer instances keyed by (model, background).
    cache_key is just used to differentiate entries; the real identity
    comes from the function arguments.
    """
    return _make_explainer(_model, background_df, feature_names)


def get_shap_values(model: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame,
                    max_bg: int = 50, max_explain: int = 50):
    """
    GLOBAL SHAP for a sample of test points, plus global weights table.

    Returns:
        shap_values: np.ndarray or list-like (per SHAP API)
        explain_df:  pd.DataFrame used for explanation
        shap_global_df: DataFrame with columns [rank, feature, abs_mean_shap, mean_shap, std_shap]
    """
    preprocess, _ = _split_pipeline(model)
    feature_names = X_train.columns.tolist()

    # Background & explain sets (cap for tractability)
    background_df = shap.sample(X_train, min(max_bg, len(X_train)), random_state=42)
    explain_df    = shap.sample(X_test,  min(max_explain, len(X_test)), random_state=42)

    cache_key = _hash_model_and_bg(model, background_df)
    explainer = _get_explainer_cached(model, background_df, feature_names, cache_key)
    explain_data = _transform_data_for_explainer(model, explain_df)
    sv = explainer(explain_data)  # shap.Explanation (preferred) or values array (fallback)

    # Get matrix of SHAP values (n, d)
    if isinstance(sv, shap._explanation.Explanation):
        V = np.asarray(sv.values)
    else:
        V = np.asarray(sv)

    # Align feature names with the SHAP value dimension to avoid shape mismatches
    feature_names_used = feature_names if V.shape[1] == len(feature_names) else _infer_feature_names(preprocess, feature_names, V.shape[1])

    # Use the same representation for plotting as the explainer received
    try:
        explain_df_for_plot = pd.DataFrame(explain_data, columns=feature_names_used)
    except Exception:
        explain_df_for_plot = explain_df
    # Ensure numeric for plotting to avoid isfinite errors on categoricals
    explain_df_for_plot = _coerce_df_to_numeric(explain_df_for_plot).apply(pd.to_numeric, errors="coerce")

    abs_mean = np.mean(np.abs(V), axis=0)
    mean_signed = np.mean(V, axis=0)
    std = np.std(V, axis=0)

    shap_global_df = (
        pd.DataFrame({
            "feature": feature_names_used,
            "abs_mean_shap": abs_mean,
            "mean_shap": mean_signed,
            "std_shap": std
        })
        .sort_values("abs_mean_shap", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )
    shap_global_df.insert(0, "rank", np.arange(1, len(shap_global_df) + 1))
    # Compatibility: some downstream code expects an 'abs_mean' column name
    shap_global_df["abs_mean"] = shap_global_df["abs_mean_shap"]

    # If we have an Explanation, replace its data with the numeric, aligned version for plotting
    if isinstance(sv, shap._explanation.Explanation):
        try:
            sv = shap.Explanation(
                values=sv.values,
                base_values=sv.base_values,
                data=explain_df_for_plot.to_numpy(dtype=float, copy=False),
                feature_names=feature_names_used,
            )
        except Exception:
            pass

    return sv, explain_df_for_plot, shap_global_df


def compute_shap_direction(shap_global_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with columns [feature, sign] where sign in {-1,0,1}
    is the sign of the MEAN SHAP value for the positive class.
    """
    return pd.DataFrame({
        "feature": shap_global_df["feature"].values,
        "sign": np.sign(shap_global_df["mean_shap"].values).astype(int)
    })


def summarize_reliability(df: pd.DataFrame, sanity_ratio: float, n_trials: int, bg_size: int) -> str:
    """
    Produce a short textual summary of SHAP reliability checks.

    Inputs:
      - df: DataFrame with columns ["feature", "avg_rank", "std_rank"]
      - sanity_ratio: float (orig_mass / randomized_mass)
      - n_trials: int
      - bg_size: int

    Returns a 1-2 paragraph string summarising top features, stability counts,
    and interpretation of the sanity ratio.
    """
    # Defensive copy and ensure expected columns exist
    if df is None or df.empty:
        return (
            f"No feature ranking data available. Ran {n_trials} trials with background size {bg_size}. "
            "Cannot summarise stability without feature ranks."
        )

    # Work on a cleaned DataFrame
    working = df.copy()
    # Ensure required columns exist
    for col in ("feature", "avg_rank", "std_rank"):
        if col not in working.columns:
            raise ValueError(f"Input df must contain column '{col}'")

    # Sort by avg_rank ascending (1 = most important)
    working = working.sort_values("avg_rank", ascending=True).reset_index(drop=True)
    top_k = min(3, len(working))
    top_df = working.iloc[:top_k]
    top_features = top_df["feature"].astype(str).tolist()

    # Describe top features' stability
    stds = top_df["std_rank"].fillna(float("inf"))
    if (stds <= 0.05).all():
        top_desc = "maintained identical ranks across all trials"
    elif (stds <= 0.3).all():
        top_desc = "showed very small rank variation"
    else:
        top_desc = (
            "showed some rank variation but remained consistently among the most important features"
        )

    # Remaining features stability counts
    remaining = working.iloc[top_k:]
    if remaining.empty:
        stable_count = moderate_count = unstable_count = 0
    else:
        rem_stds = remaining["std_rank"].fillna(float("inf"))
        stable_count = int((rem_stds < 0.5).sum())
        moderate_count = int(((rem_stds >= 0.5) & (rem_stds < 1.0)).sum())
        unstable_count = int((rem_stds >= 1.0).sum())

    # Interpret sanity_ratio
    if sanity_ratio >= 0.95:
        sanity_phrase = (
            "indicates that the explanations are driven by genuine model–data structure rather than noise."
        )
    elif sanity_ratio >= 0.85:
        sanity_phrase = (
            "suggests that the explanations capture useful structure, although some noise is present."
        )
    else:
        sanity_phrase = (
            "raises concerns that the explanations may be dominated by noise; additional trials or a larger background set may be required."
        )

    # Nicely join top features
    if len(top_features) == 1:
        top_list_str = top_features[0]
    elif len(top_features) == 2:
        top_list_str = f"{top_features[0]} and {top_features[1]}"
    else:
        top_list_str = ", ".join(top_features[:-1]) + f", and {top_features[-1]}"

    para1 = (
        f"Using {n_trials} trials and a background size of {bg_size}, the top {top_k} feature(s) by average rank were: {top_list_str}. "
        f"These {('features' if top_k>1 else 'feature')} {top_desc}."
    )

    remaining_count = len(working) - top_k
    para2 = (
        f"Among the remaining {remaining_count} feature(s), most were stable ({stable_count}), "
        f"with {moderate_count} showing moderate variation and {unstable_count} exhibiting unstable rankings. "
        f"The sanity ratio of {sanity_ratio:.2f} {sanity_phrase}"
    )

    # Simple conclusion based on sanity + instability
    if sanity_ratio >= 0.95 and unstable_count == 0:
        conclusion = (
            "Taken together, these results indicate that the SHAP explanations are stable and suitable for global interpretation."
        )
    elif sanity_ratio >= 0.85:
        conclusion = (
            "Taken together, these results suggest reasonable reliability but some caution is warranted when using these explanations for high-stakes decisions."
        )
    else:
        conclusion = (
            "Taken together, these results advise caution: consider running more trials or increasing the background size before relying on these explanations."
        )

    return f"{para1}\n\n{para2} {conclusion}"


def build_perturbation_subsets(shap_direction_df: pd.DataFrame,
                               numeric_feature_names: list[str],
                               immutable_columns: list[str] | None = None) -> dict:
    """
    Build Sup / Sdown / Simmutable as required by the paper's Stage-2 search.
    Convention: positive class = default. To reduce default risk:
      - If mean SHAP sign > 0, DECREASE the feature (? Sdown)
      - If mean SHAP sign < 0, INCREASE the feature (? Sup)
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
    preprocess, _ = _split_pipeline(model)
    background_df = shap.sample(X_train, min(max_bg, len(X_train)), random_state=42)

    cache_key = _hash_model_and_bg(model, background_df)
    explainer = _get_explainer_cached(model, background_df, feature_names, cache_key)
    instance_data = _transform_data_for_explainer(model, instance_df)
    sv = explainer(instance_data)

    # Ensure a 1D explanation for the single row
    if isinstance(sv, shap._explanation.Explanation):
        values = np.asarray(sv.values).reshape(-1)
        base_value = np.asarray(sv.base_values).reshape(-1)[0]
    else:
        values = np.asarray(sv)[0]
        base_value = explainer.expected_value if np.isscalar(explainer.expected_value) else explainer.expected_value[1]

    feature_names_used = feature_names if len(values) == len(feature_names) else _infer_feature_names(preprocess, feature_names, len(values))
    try:
        row_data = pd.Series(np.asarray(instance_data).reshape(-1)[:len(values)], index=feature_names_used)
    except Exception:
        row_data = pd.Series(np.asarray(instance_df).reshape(-1)[:len(values)], index=feature_names_used)

    return shap.Explanation(
        values=values,
        base_values=base_value,
        data=row_data,
        feature_names=feature_names_used
    )


def get_shap_values_stable(
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: Optional[pd.Series] = None,
    y_test: Optional[pd.Series] = None,
    n_trials: int = 10,
    bg_size: int = 200,
    explain_size: int = 512,
    random_state: int = 42,
):
    """
    Unified GLOBAL SHAP + Rank Stability routine.

    Computes SHAP values across multiple resamples (trials) with stratified sampling
    when labels are available. Returns:
      - sv_example: shap.Explanation for plotting
      - explain_df_example: DataFrame aligned to sv_example
      - shap_global_df: DataFrame with global SHAP stats + rank stability metrics

    Columns in shap_global_df:
      - rank, feature, abs_mean_shap, mean_shap, std_shap, avg_rank, std_rank, abs_mean
    """
    preprocess, _ = _split_pipeline(model)
    feature_names = X_train.columns.tolist()
    rng = check_random_state(random_state)

    # Accumulators
    sum_abs = None
    sum_signed = None
    sum_sq = None
    total_n = 0
    rank_list = []
    sv_example = None
    explain_df_example = None
    feature_names_used = None

    for t in range(n_trials):
        rs_t = rng.randint(1e9)

        # Sample background
        if y_train is not None:
            bg_X, _ = _stratified_sample(X_train, y_train, bg_size, rs_t)
        else:
            bg_X = X_train.sample(min(bg_size, len(X_train)), random_state=rs_t)

        # Build explainer
        explainer = _make_explainer(model, bg_X, feature_names)

        # Sample explain set
        if y_test is not None:
            ex_X, _ = _stratified_sample(X_test, y_test, explain_size, rs_t)
        else:
            ex_X = X_test.sample(min(explain_size, len(X_test)), random_state=rs_t)

        # Transform and explain
        explain_data = _transform_data_for_explainer(model, ex_X)
        sv_t = explainer(explain_data)

        # Extract values matrix
        V = np.asarray(sv_t.values) if hasattr(sv_t, "values") else np.asarray(sv_t)

        # Infer feature names
        fn_used = (
            feature_names
            if V.shape[1] == len(feature_names)
            else _infer_feature_names(preprocess, feature_names, V.shape[1])
        )
        if feature_names_used is None:
            feature_names_used = fn_used

        # Compute abs_mean for this trial and rank
        abs_mean_t = np.mean(np.abs(V), axis=0)
        ranks_t = abs_mean_t.argsort()[::-1]
        rank_map = {fn_used[i]: r for r, i in enumerate(ranks_t)}
        rank_list.append(rank_map)

        # Accumulate global stats
        if sum_abs is None:
            sum_abs = np.sum(np.abs(V), axis=0)
            sum_signed = np.sum(V, axis=0)
            sum_sq = np.sum(V**2, axis=0)
        else:
            sum_abs += np.sum(np.abs(V), axis=0)
            sum_signed += np.sum(V, axis=0)
            sum_sq += np.sum(V**2, axis=0)
        total_n += V.shape[0]

        # Store one example for plotting (first trial with Explanation)
        if sv_example is None and isinstance(sv_t, shap._explanation.Explanation):
            try:
                ex_df_plot = pd.DataFrame(explain_data, columns=fn_used)
            except Exception:
                ex_df_plot = ex_X.copy()
            ex_df_plot = _coerce_df_to_numeric(ex_df_plot).apply(
                pd.to_numeric, errors="coerce"
            )
            try:
                sv_example = shap.Explanation(
                    values=sv_t.values,
                    base_values=sv_t.base_values,
                    data=ex_df_plot.to_numpy(dtype=float, copy=False),
                    feature_names=fn_used,
                )
                explain_df_example = ex_df_plot
            except Exception:
                # Fallback if reconstruction fails
                sv_example = sv_t
                explain_df_example = ex_df_plot

    if total_n == 0:
        raise ValueError("No SHAP values computed across trials.")

    # Global statistics
    abs_mean = sum_abs / total_n
    mean_signed = sum_signed / total_n
    mean_sq = sum_sq / total_n
    std = np.sqrt(np.maximum(mean_sq - mean_signed**2, 0.0))

    # Rank stability matrix
    n_features = len(feature_names_used)
    ranks_mat = np.zeros((n_trials, n_features))
    for t, rank_map in enumerate(rank_list):
        for i, f in enumerate(feature_names_used):
            ranks_mat[t, i] = rank_map.get(f, n_features)
    avg_rank = ranks_mat.mean(axis=0)
    std_rank = ranks_mat.std(axis=0)

    # Build global DataFrame
    shap_global_df = pd.DataFrame(
        {
            "feature": feature_names_used,
            "abs_mean_shap": abs_mean,
            "mean_shap": mean_signed,
            "std_shap": std,
            "avg_rank": avg_rank,
            "std_rank": std_rank,
        }
    ).sort_values("abs_mean_shap", ascending=False, kind="mergesort").reset_index(drop=True)

    shap_global_df.insert(0, "rank", np.arange(1, len(shap_global_df) + 1))
    shap_global_df["abs_mean"] = shap_global_df["abs_mean_shap"]

    return sv_example, explain_df_example, shap_global_df
