# streamlit_app.py
from distro import name
import streamlit as st
# Prefer wide layout for the app by default
st.set_page_config(layout="wide")
import pandas as pd
import numpy as np  # Added for benchmark calculations
import altair as alt
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from sklearn.model_selection import train_test_split
from io import StringIO # Added for reading full DF
from dotenv import load_dotenv # --- NEW: Import dotenv ---
from feature_importance_paper import compute_feature_importance_for_files
import hashlib, io
from io import BytesIO
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import os
import shutil
try:
    import openai
except ImportError:
    openai = None
import re
from expand_controls import (
    EXPAND_ALL_FLAG,
    queue_expand_all,
    fire_expand_all_if_pending,
    render_generate_report_button,
)
import inspect

# Optional statistical tests
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None

# NEW imports from shap_analysis.py
from shap_analysis import (
    shap_rank_stability, model_randomization_sanity,
    find_counterfactual, CFConstraints,
    shap_top_interactions_for_tree, plot_ice_pdp
)


# --- NEW: Load .env file ---
# Make sure .env is in the same directory as streamlit_app.py
load_dotenv()

from io import BytesIO

# --- YData/Pandas Profiling support ---
try:
    from ydata_profiling import ProfileReport  # preferred
except ImportError:
    try:
        from pandas_profiling import ProfileReport  # legacy fallback
    except ImportError:
        ProfileReport = None  # the UI will warn and disable profiling

# Import the MODELS dictionary and the new run_experiment function
try:
    from models import MODELS, run_experiment, IMBLEARN_AVAILABLE
except ImportError:
    st.error("Could not find 'MODELS' dictionary, 'run_experiment' function, or 'IMBLEARN_AVAILABLE' in models.py. Please ensure they are defined.")
    MODELS = {}
    IMBLEARN_AVAILABLE = False
    def run_experiment(files, target, models, use_smote): # Added use_smote
        return {"error": "models.py not found"}

# --- MODIFIED: Import BOTH SHAP functions ---
try:
    import shap
    import matplotlib.pyplot as plt # Import matplotlib
    
    # Load the SHAP JavaScript libraries (for waterfall plot)
    shap.initjs()
    
    # Import both global and local SHAP functions
    from shap_analysis import get_shap_values, get_local_shap_explanation, summarize_reliability, get_shap_values_stable
    # --- NEW: Import LLM explanation function ---
    from llm_explain import get_llm_explanation
    
    SHAP_AVAILABLE = True
except ImportError as e:
    SHAP_AVAILABLE = False
    get_shap_values = None 
    get_local_shap_explanation = None
    get_llm_explanation = None # Add placeholder
    st.warning(f"A required library was not found. Steps 5 & 6 may be disabled. Error: {e}")


# ---- Session bootstrap: guarantee keys exist even before __init__ runs ----
_DEFAULTS = {
    "uploaded_files_map": {},
    "selected_datasets": [],
    "use_smote": False,
    "target_column": "target",
    "selected_model_groups": [],
    "selected_models": [],
    "results": {},
    "benchmark_results_df": None,
    "benchmark_auc_comparison": None,
    "run_shap": False,
    "full_dfs": {},
    "feature_selection": {},
    "fi_results_cache": {},
    "fi_signature": None,
    "fi_stale": False,
    "benchmark_requested": False,
    "ydata_profiles": {},         # { dataset_name: {"html": str, "filename": str} }
    "ydata_minimal_mode": False,  # remember the toggle choice
    "rel_n_trials": 10,
    "rel_n_bg": 200,
    "rel_speed_preset": "Balanced (10 trials / 200 bg)",
    "_rel_prev_preset": "Balanced (10 trials / 200 bg)",
    "use_stable_shap": False,
    "shap_selected_datasets": [],
    "stable_shap_trials": 10,
    "stable_shap_bg_size": 200,
    "stable_shap_explain_size": 512,
    # Reliability step defaults
    "run_reliability_test": False,
    "reliability_selected_datasets": [],
    "reliability_results": {},
    "reliability_ratios": {},
    "reliability_texts": {},
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def _df_to_named_bytesio(df, out_name: str) -> BytesIO:
    data = df.to_csv(index=False).encode("utf-8")
    bio = BytesIO(data)
    bio.seek(0)
    bio.name = out_name
    return bio


def _reset_experiment_state():
    # Clear ONLY things produced by Step 3/4 so the user must re-run.
    for k in [
        "results",
        "benchmark_results_df",
        "benchmark_auc_comparison",
        "trained_models",
        "cv_reports",
        "run_shap",
        "full_dfs",
    ]:
        st.session_state.pop(k, None)


def _df_to_named_bytesio(df, out_name: str) -> BytesIO:
    data = df.to_csv(index=False).encode("utf-8")  # produce bytes
    bio = BytesIO(data)
    bio.seek(0)
    bio.name = out_name
    return bio


def _prepare_benchmark_long(bench_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the benchmark dataframe is in long/tidy form with columns:
    ['dataset','model','metric','value']
    Accepts a wide-form `bench_df` or a pre-existing long-form and returns long-form.
    """
    if bench_df is None or bench_df.empty:
        return pd.DataFrame(columns=["dataset", "model", "metric", "value"])

    df = bench_df.copy()

    # identify dataset and model columns (common names)
    dataset_col = None
    model_col = None
    for c in df.columns:
        if c.lower() in ("dataset", "dataset_name", "datasetname"):
            dataset_col = c
            break
    for c in df.columns:
        if c.lower() in ("model", "model_name", "benchmark model", "benchmark_model"):
            model_col = c
            break

    # fallback guesses
    if dataset_col is None:
        if "Dataset" in df.columns:
            dataset_col = "Dataset"
        else:
            dataset_col = df.columns[0]
    if model_col is None:
        # try common patterns
        for guess in ("Model", "model", "Benchmark Model", "benchmark_model", "BenchmarkModel"):
            if guess in df.columns:
                model_col = guess
                break
        if model_col is None:
            # fallback to second column if available
            model_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    # Desired metrics (use these names for charts/tables)
    desired = ["AUC", "PCC", "F1", "Recall", "BS", "KS", "PG", "H"]

    # Collect available metrics from the DataFrame
    available_metrics = [c for c in df.columns if c in desired]

    # Now melt using only desired metrics that are present
    metric_candidates = [m for m in desired if m in df.columns]
    if not metric_candidates:
        # Try common alternate names / uppercase matches
        desired_upper = tuple([d.upper() for d in desired])
        metric_candidates = [c for c in df.columns if c.upper() in desired_upper]

    if not metric_candidates:
        # Nothing to plot
        return pd.DataFrame(columns=["dataset", "model", "metric", "value"])

    df_long = df.melt(id_vars=[dataset_col, model_col], value_vars=metric_candidates, var_name="metric", value_name="value")
    df_long = df_long.rename(columns={dataset_col: "dataset", model_col: "model"})
    # Normalize model names to simple labels (if needed)
    df_long["model"] = df_long["model"].astype(str)
    # Map model keys to canonical group labels (no guessing) — use exact list
    # Order matters: put more specific group labels before shorter ones
    canonical_groups = [
        "lr_reg",
        "lr",
        "adaboost",
        "Bag-CART",
        "BagNN",
        "Boost-DT",
        "RF",
        "SGB",
        "KNN",
        "XGB",
        "LGBM",
        "DL",
    ]

    def _map_to_group_label(m: str) -> str:
        if m is None:
            return m
        s = str(m).strip()
        s_norm = ''.join(ch for ch in s.lower() if ch.isalnum())
        for g in canonical_groups:
            g_norm = ''.join(ch for ch in g.lower() if ch.isalnum())
            if not g_norm:
                continue
            # Prefer exact or prefix matches (e.g., 'lr_reg_saga' -> 'lr_reg')
            if s_norm == g_norm or s_norm.startswith(g_norm) or f"_{g_norm}_" in f"_{s_norm}_" or s_norm.endswith(g_norm):
                return g
            # Fallback to substring match if nothing else matched
            if g_norm in s_norm:
                return g
        # fallback: return original model string
        return s

    df_long["model"] = df_long["model"].apply(_map_to_group_label)
    return df_long[["dataset", "model", "metric", "value"]]


def make_metric_chart(df_long: pd.DataFrame, metric_name: str):
    """
    df_long contains columns: ['dataset','model','metric','value']
    Returns an Altair chart for one metric.
    """
    # Fixed model order (use the canonical group labels you provided)
    model_order = ["lr","lr_reg","adaboost","Bag-CART","BagNN","Boost-DT","RF","SGB","KNN","XGB","LGBM","DL"]
    d = df_long[df_long["metric"] == metric_name].copy()
    if d.empty:
        # return an empty chart placeholder
        return alt.Chart(pd.DataFrame({"model":[], "value":[], "dataset":[]})).mark_line().encode()

    # Ensure model is categorical with fixed domain so x-axis order is consistent
    d["model"] = pd.Categorical(d["model"], categories=model_order, ordered=True)

    chart = (
        alt.Chart(d)
        .mark_line(point=True)
        .encode(
            x=alt.X("model:N", sort=model_order, title="Model"),
            # Show y-axis labels with four decimal places and avoid forcing zero baseline
            y=alt.Y(
                "value:Q",
                title=metric_name,
                axis=alt.Axis(format=".4f", tickCount=5),
                scale=alt.Scale(zero=False, nice=False),
            ),
            color=alt.Color("dataset:N", legend=alt.Legend(title="Dataset", orient="bottom")),
            tooltip=[alt.Tooltip("dataset:N"), alt.Tooltip("model:N"), alt.Tooltip("value:Q", format=".4f")],
        )
        .properties(height=250)
    )
    return chart


def _wilcoxon_abs_error_test(y_true, p1, p2):
    """Wilcoxon on per-sample absolute error; returns (stat, p, med_diff)."""
    if not SCIPY_AVAILABLE:
        return None, None, None
    y_true = np.asarray(y_true).astype(float)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    if y_true.shape[0] != p1.shape[0] or p1.shape[0] != p2.shape[0]:
        return None, None, None
    diff = np.abs(p1 - y_true) - np.abs(p2 - y_true)
    if np.allclose(diff, 0):
        return 0.0, 1.0, 0.0
    try:
        stat, p = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
        return float(stat), float(p), float(np.median(diff))
    except Exception:
        return None, None, None


def _mcnemar_test(y_true, pred1, pred2):
    """McNemar's test with continuity correction; returns (b, c, chi2, p)."""
    if not SCIPY_AVAILABLE:
        return None, None, None, None
    y_true = np.asarray(y_true).astype(int)
    p1 = np.asarray(pred1).astype(int)
    p2 = np.asarray(pred2).astype(int)
    if y_true.shape[0] != p1.shape[0] or p1.shape[0] != p2.shape[0]:
        return None, None, None, None
    b = int(np.sum((p1 == y_true) & (p2 != y_true)))
    c = int(np.sum((p1 != y_true) & (p2 == y_true)))
    if b + c == 0:
        return b, c, None, None
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_val = float(stats.chi2.sf(chi2, df=1))
    return b, c, float(chi2), p_val


def _compute_midrank(x):
    x = np.asarray(x)
    idx = np.argsort(x)
    sorted_x = x[idx]
    ranks = np.zeros(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        mid = 0.5 * (i + j - 1) + 1
        ranks[i:j] = mid
        i = j
    out = np.empty(len(x), dtype=float)
    out[idx] = ranks
    return out


def _fast_delong(predictions_sorted_transposed, label_1_count):
    """
    Fast DeLong algorithm; predictions_sorted_transposed shape = (n_classifiers, n_examples)
    label_1_count is #positives after sorting by first classifier descending.
    """
    m = int(label_1_count)
    n = predictions_sorted_transposed.shape[1] - m
    pos = predictions_sorted_transposed[:, :m]
    neg = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty((k, m))
    ty = np.empty((k, n))
    for r in range(k):
        tx[r] = _compute_midrank(pos[r])
        ty[r] = _compute_midrank(neg[r])

    tz = _compute_midrank(predictions_sorted_transposed[0])
    aucs = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)

    v10 = (tz[:m] - tx[0]) / n
    v01 = 1.0 - (tz[m:] - ty[0]) / m
    sx = np.cov(v10)
    sy = np.cov(v01)
    s = sx / m + sy / n
    return np.array([aucs]) if np.isscalar(aucs) else aucs, s


def delong_roc_test(y_true, scores1, scores2):
    """Returns dict with AUCs, diff, variance, z, p_value using DeLong."""
    if not SCIPY_AVAILABLE:
        return None
    y_true = np.asarray(y_true).astype(int)
    scores1 = np.asarray(scores1, dtype=float)
    scores2 = np.asarray(scores2, dtype=float)
    if y_true.ndim != 1 or scores1.shape[0] != y_true.shape[0] or scores2.shape[0] != y_true.shape[0]:
        return None
    if len(np.unique(y_true)) < 2:
        return None

    order = np.argsort(-scores1)
    labels_sorted = y_true[order]
    preds_sorted = np.vstack([scores1, scores2])[:, order]
    label_1_count = int(labels_sorted.sum())
    aucs, cov = _fast_delong(preds_sorted, label_1_count)

    # If DeLong collapses to a single AUC (degenerate), fall back to simple AUCs without p-value
    if np.asarray(aucs).shape[0] < 2:
        try:
            auc1 = float(roc_auc_score(y_true, scores1))
            auc2 = float(roc_auc_score(y_true, scores2))
        except Exception:
            return None
        return {
            "auc1": auc1,
            "auc2": auc2,
            "auc_diff": auc1 - auc2,
            "var": None,
            "z": None,
            "p_value": None,
        }

    # Normalize covariance to 2x2 to avoid shape/Index errors on degenerate cases
    cov = np.atleast_2d(cov)
    if cov.shape != (2, 2):
        full = np.zeros((2, 2))
        r, c = cov.shape
        full[:r, :c] = cov
        cov = full
    diff = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    if var <= 0:
        z = np.inf
        p = 0.0
    else:
        z = diff / np.sqrt(var)
        p = 2 * stats.norm.sf(abs(z))
    return {
        "auc1": float(aucs[0]),
        "auc2": float(aucs[1]),
        "auc_diff": diff,
        "var": var,
        "z": float(z),
        "p_value": float(p),
    }


def _compute_target_value_counts(
    fileobj,
    target_col: str,
    chunk_size: int = 100_000,
) -> Tuple[Optional[pd.Series], Optional[str], bool]:
    """
    Returns (series, error_message, missing_column_flag) for value counts of the target column.
    Chunked reads keep memory usage manageable on very large files.
    """
    if not fileobj:
        return None, "Missing file object.", False

    original_pos = None
    try:
        original_pos = fileobj.tell()
    except Exception:
        original_pos = None

    try:
        try:
            fileobj.seek(0)
        except Exception:
            pass

        header = pd.read_csv(fileobj, nrows=0)
        columns = header.columns.tolist()
        if target_col not in columns:
            return None, None, True

        try:
            fileobj.seek(0)
        except Exception:
            pass

        counts: Dict[Any, int] = {}
        for chunk in pd.read_csv(fileobj, usecols=[target_col], chunksize=chunk_size):
            vc = chunk[target_col].value_counts(dropna=False)
            for val, count in vc.items():
                counts[val] = counts.get(val, 0) + int(count)

        series = pd.Series(counts).sort_values(ascending=False) if counts else pd.Series(dtype="int64")
        return series, None, False
    except Exception as exc:
        return None, str(exc), False
    finally:
        if original_pos is not None:
            try:
                fileobj.seek(original_pos)
            except Exception:
                pass


def _prepare_features_for_smote_preview(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert categorical/text columns into numeric encodings so SMOTE can run.
    Mirrors the ColumnTransformer + OneHotEncoder used later, albeit simplified.
    """
    if df.empty:
        return df

    work = df.copy()

    cat_cols = work.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        # Include NaNs as their own column so we don't drop information
        work = pd.get_dummies(work, columns=cat_cols, dummy_na=True)

    bool_cols = work.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        work[bool_cols] = work[bool_cols].astype(int)

    non_numeric = work.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        work[non_numeric] = work[non_numeric].apply(pd.to_numeric, errors="coerce")

    work = work.fillna(0)
    return work


def _preview_smote_balance(
    fileobj,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Optional[pd.Series], Optional[str], bool]:
    """
    Mimics the training split + SMOTE application used later so we can show
    the balanced class counts that the models will actually see.
    """
    if not IMBLEARN_AVAILABLE:
        return None, "SMOTE preview requires 'imbalanced-learn'.", False

    original_pos = None
    try:
        original_pos = fileobj.tell()
    except Exception:
        original_pos = None

    try:
        try:
            fileobj.seek(0)
        except Exception:
            pass

        df = pd.read_csv(fileobj)

        if target_col not in df.columns:
            return None, None, True

        if df[target_col].nunique(dropna=False) < 2:
            return None, "Target column must contain at least two classes for SMOTE.", False

        X = df.drop(columns=[target_col])
        y = df[target_col]

        X = _prepare_features_for_smote_preview(X)
        if X.shape[1] == 0:
            return None, "No usable feature columns for SMOTE preview.", False

        try:
            X_train, _, y_train, _ = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y,
            )
        except ValueError as exc:
            return None, str(exc), False

        try:
            from imblearn.over_sampling import SMOTE

            smoter = SMOTE(random_state=random_state)
            _, y_balanced = smoter.fit_resample(X_train, y_train)
        except Exception as exc:
            return None, str(exc), False

        counts = y_balanced.value_counts().sort_values(ascending=False)
        return counts, None, False
    except Exception as exc:
        return None, f"SMOTE preview failed: {exc}", False
    finally:
        if original_pos is not None:
            try:
                fileobj.seek(original_pos)
            except Exception:
                pass


@st.cache_data(show_spinner=False)
def _generate_profile_html(df: pd.DataFrame, title: str, minimal: bool) -> str:
    """
    Build a profiling report and return it as HTML.
    Caches by DataFrame content hash, title, and minimal flag.
    """
    if ProfileReport is None:
        raise ImportError(
            "Profiling library not found. Install 'ydata-profiling' (recommended) "
            "or 'pandas-profiling'."
        )

    kwargs = {"title": title}
    if minimal:
        kwargs["minimal"] = True

    try:
        profile = ProfileReport(df, **kwargs)
    except TypeError:
        # Older releases may not accept 'minimal' kwarg
        kwargs.pop("minimal", None)
        profile = ProfileReport(df, **kwargs)
        if minimal:
            # Try toggling via config when supported
            try:
                profile.config.set_option("minimal", True)
            except Exception:
                pass

    # Prefer richer layout when supported
    try:
        profile.config.set_option("explorative", True)
    except Exception:
        pass

    return profile.to_html()


def _bytesig_of_upload(fobj) -> str:
    """
    Compute a stable short hash signature of an uploaded file's content
    without destroying its read pointer.
    Used by Step 1.25 to detect if inputs changed.
    """
    try:
        pos = fobj.tell()
    except Exception:
        pos = None
    try:
        # If it's an UploadedFile, it may expose getvalue()
        if hasattr(fobj, "getvalue"):
            data = fobj.getvalue()
        else:
            data = fobj.read()
    finally:
        try:
            if pos is not None:
                fobj.seek(pos)
        except Exception:
            pass

    if not isinstance(data, (bytes, bytearray)):
        data = bytes(str(data), "utf-8")

    return hashlib.md5(data).hexdigest()


class ExperimentSetupApp:
    """
    A class to encapsulate the Streamlit experiment setup wizard.
    """
    
    def __init__(self):
        """
        Initialize the app and set the page title.
        """
        st.title("Elucidate")
        
        # Initialize session state if it doesn't exist
        if 'uploaded_files_map' not in st.session_state:
            st.session_state.uploaded_files_map = {} # Stores the actual UploadedFile objects
        if 'selected_datasets' not in st.session_state:
            st.session_state.selected_datasets = [] # Stores just the names
        
        if 'use_smote' not in st.session_state:
            st.session_state.use_smote = False
            
        if 'target_column' not in st.session_state:
            st.session_state.target_column = "target"
        if 'selected_model_groups' not in st.session_state:
            st.session_state.selected_model_groups = []
        if 'selected_models' not in st.session_state:
            st.session_state.selected_models = []
        
        if 'results' not in st.session_state:
            st.session_state.results = {} 
        
        if 'benchmark_results_df' not in st.session_state:
            st.session_state.benchmark_results_df = None # Will store a DataFrame
        if 'benchmark_auc_comparison' not in st.session_state:
            st.session_state.benchmark_auc_comparison = None 

        if 'run_shap' not in st.session_state:
            st.session_state.run_shap = False
            
        # --- NEW: Cache for full DataFrames for Step 6 ---
        if 'full_dfs' not in st.session_state:
            st.session_state.full_dfs = {}

        if 'feature_selection' not in st.session_state:
            # { dataset_name: [list of selected feature columns] }
            st.session_state.feature_selection = {}

        if "fi_results_cache" not in st.session_state:
            # { dataset_name: payload }, same structure you already display (rf/lr/merged/meta)
            st.session_state.fi_results_cache = {}

        if "fi_signature" not in st.session_state:
            # tuple that identifies what the cache corresponds to (files+target)
            st.session_state.fi_signature = None

        # run once early in the app
        def ds_key(name):
            return name.replace('.csv', '').replace(' ', '_')
            
        if "benchmarks" in st.session_state:
            for k in list(st.session_state.benchmarks.keys()):
                nk = ds_key(k)
                if nk != k and nk not in st.session_state.benchmarks:
                    st.session_state.benchmarks[nk] = st.session_state.benchmarks.pop(k)

        if "fi_results_cache" not in st.session_state:
            st.session_state.fi_results_cache = {}
        if "fi_signature" not in st.session_state:
            st.session_state.fi_signature = None
        if "fi_stale" not in st.session_state:
            st.session_state.fi_stale = False


    def _on_preprocessing_change(self):
            """Resets results if preprocessing options change."""
            st.session_state.results = {}
            st.session_state.benchmark_results_df = None
            st.session_state.benchmark_auc_comparison = None
            st.session_state.run_shap = False 
            st.session_state.full_dfs = {} # --- NEW: Clear DF cache ---

            # --- NEW: also clear feature-importance cache/state ---
            st.session_state.fi_results_cache = {}
            st.session_state.fi_signature = None
            st.session_state.fi_stale = False
            
            # --- ADD THIS LINE ---
            st.session_state.pop("global_shap_dfs", None)


    def _render_step_1_dataset_selection(self):
        """
        Renders the UI for dataset selection (Step 1).
        """
        st.header("Step 1: Select Datasets")
        
        with st.expander("Upload datasets in experiment", expanded=True):
            uploads = st.file_uploader(
                "Upload CSV files:", type="csv", accept_multiple_files=True
            )

            if uploads:
                new_file_names = [f.name for f in uploads]
                # Check if files have actually changed before clearing results
                if set(new_file_names) != set(st.session_state.selected_datasets):
                    # Use the callback to clear all results
                    self._on_preprocessing_change()

                # Store the actual file objects in a map
                st.session_state.uploaded_files_map = {f.name: f for f in uploads}
                # Store just the names for display and selection
                st.session_state.selected_datasets = new_file_names
            
            else:
                # Clear state if all files are removed
                if st.session_state.selected_datasets: # Only clear if there *were* files
                    self._on_preprocessing_change()
                st.session_state.uploaded_files_map = {}
                st.session_state.selected_datasets = []


    def _display_step_1_results(self):
        """
        Displays the results from Step 1 based on session state.
        """
        if st.session_state.get("selected_datasets"):
            st.success(f"Datasets selected: {', '.join(st.session_state.selected_datasets)}")
        else:
            st.info("No datasets selected.")

        # Show 5 sample rows for each uploaded dataset (if available)
        if st.session_state.uploaded_files_map:
            for name in st.session_state.selected_datasets:
                fileobj = st.session_state.uploaded_files_map.get(name)
                if not fileobj:
                    continue
                # Put sample + counts inside a collapsed per-dataset expander
                with st.expander(f"Preview: {name}", expanded=False):
                    try:
                        # Always rewind before each read
                        try: 
                            fileobj.seek(0)
                        except Exception:
                            pass

                        # ---- 0) Sample preview (first 5 rows)
                        df_head = pd.read_csv(fileobj, nrows=5)
                        st.markdown("**Sample (first 5 rows)**")
                        st.dataframe(df_head)

                        # ---- 1) Robust SHAPE (rows, cols) without loading full file
                        # Read header for columns
                        try:
                            fileobj.seek(0)
                        except Exception:
                            pass
                        header = pd.read_csv(fileobj, nrows=0)
                        cols = header.columns.tolist()
                        n_cols = len(cols)

                        # Count rows via chunked pass on any one column
                        CHUNK = 200_000
                        try:
                            fileobj.seek(0)
                        except Exception:
                            pass
                        n_rows = 0
                        for chunk in pd.read_csv(fileobj, usecols=[cols[0]] if cols else None,
                                                chunksize=CHUNK):
                            n_rows += len(chunk)

                        st.markdown(f"**Shape:** ({n_rows:,}, {n_cols:,})")

                        # ---- 2) Info-style table (dtype + non-null counts), computed in chunks
                        if cols:
                            # Accumulators
                            non_null = {c: 0 for c in cols}
                            dtypes_seen = None

                            try:
                                fileobj.seek(0)
                            except Exception:
                                pass
                            for chunk in pd.read_csv(fileobj, chunksize=CHUNK):
                                # dtypes from first chunk are good enough in practice
                                if dtypes_seen is None:
                                    dtypes_seen = chunk.dtypes
                                # accumulate non-null counts
                                nn = chunk.notna().sum()
                                for c in nn.index:
                                    non_null[c] += int(nn[c])

                            info_df = pd.DataFrame({
                                "column": cols,
                                "non_null": [non_null[c] for c in cols],
                                "nulls": [n_rows - non_null[c] for c in cols],
                                "%_non_null": [
                                    (non_null[c] / n_rows * 100.0) if n_rows else float("nan")
                                    for c in cols
                                ],
                                "dtype": [str(dtypes_seen.get(c, "object")) if dtypes_seen is not None else "unknown"
                                        for c in cols],
                            })
                            # nicer sorting: non-null desc, then name
                            info_df = info_df.sort_values(by=["non_null", "column"], ascending=[False, True], ignore_index=True)
                            st.markdown("**Info (concise):**")
                            st.dataframe(info_df)

                        # ---- 3) Describe (bounded sample for safety)
                        DESC_ROWS = 50_000  # adjust if you want more/less fidelity vs speed
                        try:
                            fileobj.seek(0)
                        except Exception:
                            pass
                        df_desc_sample = pd.read_csv(fileobj, nrows=DESC_ROWS)
                        st.markdown(f"**Describe() on first {min(DESC_ROWS, n_rows):,} rows (numeric columns):**")
                        st.dataframe(df_desc_sample.describe(include='number').round(6))

                        # ---- 4) Optional: Pairplot (scatter matrix) for numeric columns (bounded)
                        try:
                            num_cols = df_desc_sample.select_dtypes(include=["number"]).columns.tolist()
                        except Exception:
                            num_cols = []

                        st.markdown("**Pairplot (scatter matrix)**")
                        if not num_cols:
                            st.info("No numeric columns available for pairplot.")
                        else:
                            # Suggest up to 8 columns by default
                            max_pair_cols = 8
                            suggested = num_cols[:max_pair_cols]
                            key_cols = f"pairplot_cols_{name}"
                            selected_pair_cols = st.multiselect(
                                "Choose numeric columns for pairplot (max 8):",
                                options=num_cols,
                                default=suggested,
                                key=key_cols,
                                help="Select a subset of numeric columns to visualize. Pairplot is limited to 8 columns for performance.",
                            )

                            if selected_pair_cols:
                                if len(selected_pair_cols) > max_pair_cols:
                                    st.warning(f"Please select at most {max_pair_cols} columns. Currently selected: {len(selected_pair_cols)}")
                                else:
                                    btn_key = f"btn_pairplot_{name}"
                                    if st.button("Show pairplot (sampled)", key=btn_key):
                                        with st.spinner("Rendering pairplot (this may take a few seconds)..."):
                                            try:
                                                # Try to read a bounded sample directly from the file for memory safety
                                                try:
                                                    fileobj.seek(0)
                                                except Exception:
                                                    pass
                                                max_rows_pp = 2000
                                                try:
                                                    df_pp = pd.read_csv(fileobj, usecols=selected_pair_cols, nrows=max_rows_pp)
                                                except Exception:
                                                    # Fallback to slicing the describe/sample frame
                                                    df_pp = df_desc_sample[selected_pair_cols].sample(min(len(df_desc_sample), max_rows_pp), random_state=0)

                                                if df_pp is None or df_pp.empty:
                                                    st.warning("No data available to plot.")
                                                else:
                                                    try:
                                                        import seaborn as sns
                                                        import matplotlib.pyplot as plt
                                                        pp = sns.pairplot(df_pp)
                                                        st.pyplot(pp.fig)
                                                        plt.close(pp.fig)
                                                    except Exception as e_pp:
                                                        st.error(f"Pairplot failed: {e_pp}")
                                            except Exception as e:
                                                st.error(f"Could not prepare pairplot: {e}")

                    except Exception as e:
                        st.warning(f"Could not produce preview for {name}: {e}")

                    target = st.session_state.get('target_column', 'target')
                    counts_series, err, missing = _compute_target_value_counts(fileobj, target)
                    if missing:
                        st.info(f"Target column '{target}' not found in this file.")
                    elif err:
                        st.warning(f"Could not compute value counts for {name}: {err}")
                    elif counts_series is not None:
                        st.markdown(f"**Value counts (`{target}`)**")
                        st.write(counts_series.to_frame(name="count"))


    def _render_step_1_5_preprocessing_options(self):
        """
        Renders the UI for preprocessing options (Step 1.5).
        """
        st.header("Step 1.5: Preprocessing Options")
        
        # Disable checkbox if imblearn is not installed
        smote_disabled = not IMBLEARN_AVAILABLE
        
        st.session_state.use_smote = st.checkbox(
            "Apply SMOTE (Synthetic Minority Over-sampling TEchnique)",
            value=st.session_state.use_smote,
            on_change=self._on_preprocessing_change,
            disabled=smote_disabled,
            help="If checked, SMOTE will be applied to the *training data* to handle class imbalance before model fitting. Requires 'imbalanced-learn' to be installed."
        )
        
        if smote_disabled:
            st.warning("SMOTE is disabled because the 'imbalanced-learn' library was not found. Please install it to enable this feature.")


    def _render_step_1_4_feature_selector(self):
        """
        Step 1.4: Let the user choose independent variables per dataset.
        Default = all columns except the target.
        """
        st.header("Step 1.4: Select Independent Variables (per dataset)")

        if not st.session_state.selected_datasets or not st.session_state.uploaded_files_map:
            st.info("Upload datasets in Step 1 to choose features.")
            return

        target = st.session_state.get("target_column", "target")

        for name in st.session_state.selected_datasets:
            fileobj = st.session_state.uploaded_files_map.get(name)
            if not fileobj:
                continue

            with st.expander(f"Choose features for: {name}", expanded=False):
                # ----------------------------------------------------------
                # 1️⃣ Derive candidate features for this dataset
                # ----------------------------------------------------------
                import pandas as pd

                target = st.session_state.get("target_column", "target")
                fobj = st.session_state.uploaded_files_map.get(name)

                try:
                    fobj.seek(0)
                    header_df = pd.read_csv(fobj, nrows=0)
                    all_cols = list(header_df.columns)
                except Exception as e:
                    st.warning(f"Could not read columns for {name}: {e}")
                    try:
                        fobj.seek(0)
                        all_cols = list(pd.read_csv(fobj, nrows=100).columns)
                    except Exception as e2:
                        st.error(f"Fallback read failed: {e2}")
                        all_cols = []

                # Drop any empty or unnamed columns
                all_cols = [c for c in all_cols if not str(c).startswith("Unnamed:")]

                # Exclude target column if present
                if target in all_cols:
                    candidate_features = [c for c in all_cols if c != target]
                else:
                    candidate_features = all_cols[:]

                # Deduplicate cleanly
                seen = set()
                candidate_features = [c for c in candidate_features if not (c in seen or seen.add(c))]

                # ----------------------------------------------------------
                # 2️⃣ Stable multiselect (default = all selected)
                # ----------------------------------------------------------
                key = f"feature_select_{name}"
                store_key = "feature_selection"

                # Initialize top-level store if missing
                if store_key not in st.session_state:
                    st.session_state[store_key] = {}

                # Initialize this dataset’s widget only once
                if key not in st.session_state:
                    # Start with all columns selected by default
                    st.session_state[key] = candidate_features[:]
                    st.session_state[store_key][name] = st.session_state[key][:]

                # ---------- Quick-select from Feature Importance (if available) ----------
                fi_cache = st.session_state.get("fi_results_cache", {})
                fi_payload = fi_cache.get(name)

                if fi_payload:
                    src_choice = st.radio(
                        "Feature-importance source",
                        ["Merged (RF/L1-LR)", "RandomForest only", "L1-LR only"],
                        horizontal=True,
                        key=f"fi_src_{name}",
                        help="Use the ranking produced in Step 1.25."
                    )

                    topn_choice = st.selectbox(
                        "Quick-select top features",
                        ["—", "Top 5", "Top 10", "Top 15", "Top 20"],
                        key=f"fi_topn_{name}",
                        help="Applies to the multiselect below. Re-runs of Step 3 are required."
                    )

                    # Build ranked list according to source
                    try:
                        if src_choice.startswith("Merged"):
                            ranked = list(fi_payload["merged"]["feature"])
                        elif src_choice.startswith("RandomForest"):
                            # assume 'rf' table is already sorted by importance desc
                            ranked = list(fi_payload["rf"]["feature"])
                        else:  # L1-LR only
                            # assume 'lr' table has absolute-coef ranking
                            ranked = list(fi_payload["lr"]["feature"])
                    except Exception:
                        ranked = []

                    # Filter to columns actually present in this dataset and not the target
                    ranked = [c for c in ranked if c in candidate_features]

                    # If user picked a Top-N, apply it to the multiselect value and reset results
                    top_lookup = {"Top 5": 5, "Top 10": 10, "Top 15": 15, "Top 20": 20}
                    if topn_choice in top_lookup and ranked:
                        N = top_lookup[topn_choice]
                        topN = ranked[:N]

                        # write into the multiselect's session key (seeded below)
                        mk = f"feature_select_{name}"
                        st.session_state[mk] = topN[:]  # overwrite selection

                        # mirror into canonical store
                        st.session_state["feature_selection"][name] = topN[:]

                        # changing features must invalidate downstream results
                        _reset_experiment_state()
                        st.info(f"Applied {topn_choice} from {src_choice}. Step 3 results reset.")
                else:
                    st.caption("Compute Step 1.25 first to enable Top-N quick-select.")



                # “Select all” / “Clear” buttons
                c1, c2, _ = st.columns([1, 1, 6])
                with c1:
                    if st.button("Select all", key=f"selall_{name}"):
                        st.session_state[key] = candidate_features[:]
                with c2:
                    if st.button("Clear", key=f"clear_{name}"):
                        st.session_state[key] = []

                # Multiselect reads and writes directly to its stable key
                sel = st.multiselect(
                    "Select independent variables (used downstream):",
                    options=candidate_features,
                    key=key,
                    help="All columns selected by default. Use buttons above to change selections.",
                )

                # Mirror the value into canonical store
                st.session_state[store_key][name] = list(sel)


    def _display_step_1_5_results(self):
        """
        Displays the results from Step 1.5 based on session state.
        """
        smote_enabled = st.session_state.use_smote
        target = st.session_state.get("target_column", "target")

        if smote_enabled:
            st.info("SMOTE (Oversampling) is **Enabled**.")
        else:
            st.info("SMOTE (Oversampling) is **Disabled**.")

        datasets = st.session_state.get("selected_datasets", [])
        files_map = st.session_state.get("uploaded_files_map", {})
        if not datasets or not files_map:
            st.caption("Upload datasets in Step 1 to inspect the current target distribution.")
            return

        st.markdown(f"**Current value counts for `{target}`**")

        for name in datasets:
            fileobj = files_map.get(name)
            if not fileobj:
                st.warning(f"Uploaded file for '{name}' is unavailable.")
                continue

            counts_series, err, missing = _compute_target_value_counts(fileobj, target)
            st.markdown(f"Dataset: `{name}`")
            if missing:
                st.info(f"Target column '{target}' not found in this file.")
                continue
            if err:
                st.warning(f"Could not compute value counts for {name}: {err}")
                continue
            if counts_series is None:
                st.info("No rows available to summarize.")
                continue

            counts_df = (
                counts_series.to_frame(name="count")
                .rename_axis("value")
                .reset_index()
            )
            total = counts_df["count"].sum()
            if total > 0:
                counts_df["percent"] = (counts_df["count"] / total).map(lambda x: f"{x:.2%}")
            else:
                counts_df["percent"] = "-"

            st.caption("Raw dataset distribution")
            st.dataframe(counts_df, use_container_width=True)

            if smote_enabled and IMBLEARN_AVAILABLE:
                sm_counts, sm_err, _ = _preview_smote_balance(fileobj, target)
                if sm_err:
                    st.warning(f"SMOTE-balanced preview for {name} failed: {sm_err}")
                elif sm_counts is not None:
                    sm_df = (
                        sm_counts.to_frame(name="count")
                        .rename_axis("value")
                        .reset_index()
                    )
                    sm_total = sm_df["count"].sum()
                    if sm_total > 0:
                        sm_df["percent"] = (sm_df["count"] / sm_total).map(lambda x: f"{x:.2%}")
                    else:
                        sm_df["percent"] = "-"
                    st.caption("Training split after SMOTE (80/20 stratified)")
                    st.dataframe(sm_df, use_container_width=True)

        if smote_enabled and IMBLEARN_AVAILABLE:
            st.caption(
                "Raw counts use the entire dataset. SMOTE preview applies the same 80/20 stratified split and SMOTE(random_state=42) that Step 3 uses for training."
            )
        elif smote_enabled:
            st.caption("SMOTE preview unavailable because the 'imbalanced-learn' dependency is missing.")
        else:
            st.caption(
                "Counts reflect the raw dataset before splitting. Enable SMOTE to preview the balanced training distribution."
            )


    def _render_step_1_3_ydata_profiles(self):
        """
        Step 1.3: Generate YData (pandas) profiling reports for one or more datasets.
        Produces embedded previews and per-dataset HTML downloads.
        """
        st.header("Step 1.3: Data Profiling (YData)")

        if ProfileReport is None:
            st.error(
                "Profiling library not available. Install `ydata-profiling` "
                "(preferred) or `pandas-profiling` to enable this step."
            )
            return

        if not st.session_state.selected_datasets or not st.session_state.uploaded_files_map:
            st.info("Upload datasets in Step 1 to enable profiling.")
            return

        # --- Controls ---
        # multiselect: choose which datasets to profile (default = all currently selected)
        ds_to_profile = st.multiselect(
            "Choose datasets to profile:",
            options=st.session_state.selected_datasets,
            default=st.session_state.selected_datasets,
            key="ydata_ds_select",
            help="You can profile multiple datasets at once."
        )

        st.session_state.ydata_minimal_mode = st.toggle(
            "Use minimal mode (faster on large files)",
            value=st.session_state.get("ydata_minimal_mode", False),
            key="ydata_minimal_mode_toggle",
        )

        c1, c2 = st.columns([1, 3])
        with c1:
            run_clicked = st.button("Generate profiling reports", type="primary", key="btn_ydata_profile")
        with c2:
            st.caption("Reports are built on the entire file. For very large CSVs, enable minimal mode.")

        # --- Build reports when requested ---
        if run_clicked and ds_to_profile:
            with st.spinner("Building profiling reports..."):
                for name in ds_to_profile:
                    fobj = st.session_state.uploaded_files_map.get(name)
                    if not fobj:
                        st.warning(f"File object for '{name}' not found; skipping.")
                        continue

                    # Always rewind before each read
                    try:
                        fobj.seek(0)
                    except Exception:
                        pass

                    try:
                        # Load the full DataFrame for the profiling run
                        df_full = pd.read_csv(fobj)
                    except Exception as exc:
                        st.error(f"Could not read '{name}' for profiling: {exc}")
                        continue

                    try:
                        html = _generate_profile_html(
                            df_full, title=f"{name} — Profile", minimal=st.session_state.ydata_minimal_mode
                        )
                        out_name = f"{Path(name).stem}.html"
                        st.session_state.ydata_profiles[name] = {"html": html, "filename": out_name}
                    except Exception as exc:
                        st.error(f"Failed to create profile for '{name}': {exc}")
                        continue

                    # Store in session for display & download
                    out_name = Path(name).with_suffix(".html").name
                    st.session_state.ydata_profiles[name] = {
                        "html": html,
                        "filename": out_name,
                    }

            if ds_to_profile:
                st.success("Profiling complete.")

        # --- Display any cached/built reports with download buttons ---
        if st.session_state.ydata_profiles:
            st.subheader("Profiles")
            for name in st.session_state.selected_datasets:
                prof = st.session_state.ydata_profiles.get(name)
                if not prof:
                    continue

                with st.expander(f"Profile: {name}", expanded=False):
                    # Download button
                    st.download_button(
                        "Download HTML report",
                        data=prof["html"].encode("utf-8"),
                        file_name=prof["filename"],
                        mime="text/html",
                        key=f"dl_{name}",
                    )
                    # Embedded preview
                    st.components.v1.html(prof["html"], height=600, scrolling=True)

            
    def _render_step_2_model_selection(self):
        """
        Renders the UI for model and target selection (Step 2).
        """
        st.header("Step 2: Select Models & Target")
        
        # ---------- Stable “Select model groups to run” (seed once, never snap-back) ----------
        available_model_groups = list(MODELS.keys())
        group_key = "selected_model_groups"

        # Seed exactly once (default = all groups). Do NOT reseed when empty.
        if group_key not in st.session_state:
            st.session_state[group_key] = available_model_groups[:]

        # Buttons that modify only this state
        c1, c2, _ = st.columns([1, 1, 6])
        with c1:
            if st.button("Select all model groups", key="selall_model_groups"):
                st.session_state[group_key] = available_model_groups[:]
                _reset_experiment_state()   # changing selection resets Step 3
        with c2:
            if st.button("Clear model groups", key="clear_model_groups"):
                st.session_state[group_key] = []
                _reset_experiment_state()   # changing selection resets Step 3

        # Reset Step 3 WHENEVER user changes the multiselect value
        def _on_model_groups_change():
            _reset_experiment_state()

        selected_groups = st.multiselect(
            "Select model groups to run:",
            options=available_model_groups,
            key=group_key,
            on_change=_on_model_groups_change,   # <— the crucial line
            help="All groups are selected on first load. Any change resets the Run Experiment results."
        )

        # (Optional) flatten to individual models for downstream use
        flat_model_list = []
        for g in selected_groups:
            flat_model_list.extend(MODELS.get(g, {}).keys())
        st.session_state.selected_models = flat_model_list


    def _display_step_2_results(self):
        """
        Displays the results from Step 2 based on session state.
        """
        if not st.session_state.get("target_column"):
            st.warning("Please enter a target column name.")
        else:
            st.success(f"Target column: '{st.session_state.get('target_column')}'")
        
        if st.session_state.get("selected_models"):
            st.success(f"Models to run: {', '.join(st.session_state.get('selected_models'))}")
        else:
            st.info("No models selected.")


    def _render_step_3_run_experiment(self):
        """
        Renders the "Run Experiment" button ONLY if results do not exist.
        The button's logic is contained here.
        """
        st.header("Step 3: Run Experiment")
        
        # Only show the button if the experiment hasn't been run yet
        if not st.session_state.get("results"):
            if st.button("Run Experiment"):
                # Clear any old benchmark results
                self._on_preprocessing_change() # Use this to clear everything
                
                with st.spinner("Running models on all datasets... This may take a moment."):
                    try:
                        # Get the list of actual UploadedFile objects to run on
                        # files_to_run = [
                        #     st.session_state.uploaded_files_map[name] 
                        #     for name in st.session_state.selected_datasets
                        # ]

                        # Get the list of actual UploadedFile objects to run on
                        # files_to_run = [ st.session_state.uploaded_files_map[name] for name in st.session_state.selected_datasets ]

                        # NEW: build filtered, in-memory CSVs based on feature selection
                        filtered_files = []
                        target = st.session_state.get("target_column", "target")

                        for name in st.session_state.selected_datasets:
                            fileobj = st.session_state.uploaded_files_map[name]
                            try:
                                try: fileobj.seek(0)
                                except Exception: pass
                                df_full = pd.read_csv(fileobj)

                                selected_feats = st.session_state.feature_selection.get(name)
                                if selected_feats is None or len(selected_feats) == 0:
                                    # default to all non-target columns if user didn’t select
                                    selected_feats = [c for c in df_full.columns if c != target]

                                cols_to_keep = [c for c in selected_feats if c in df_full.columns]
                                # Ensure target is present if available
                                if target in df_full.columns:
                                    cols_to_keep = cols_to_keep + [target]

                                df_reduced = df_full[cols_to_keep].copy()

                                # Keep a copy for later steps (SHAP/local analysis)
                                if 'full_dfs' in st.session_state:
                                    st.session_state.full_dfs[name] = df_reduced

                                # Serialize to CSV in-memory
                                out_name = name                      # <- do not append "__selected"
                                filtered_files.append(_df_to_named_bytesio(df_reduced, out_name))

                            except Exception as e:
                                st.error(f"Failed to apply feature selection to {name}: {e}")

                        # Pass filtered_files instead of raw uploads
                        files_to_run = filtered_files



                        # Create the dictionary of selected model groups to pass
                        selected_groups_dict = {
                            group: MODELS[group] 
                            for group in st.session_state.selected_model_groups
                            if group in MODELS
                        }
                        
                        st.session_state.results = run_experiment(
                            files_to_run,
                            st.session_state.target_column,
                            selected_groups_dict,
                            st.session_state.use_smote 
                        )
                        
                        # Check for global errors (e.g., SMOTE library missing)
                        if "error" in st.session_state.results:
                            st.error(st.session_state.results["error"])
                            st.session_state.results = {} # Clear the error
                        else:
                            st.success("Experiment complete!")
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"An error occurred during the experiment: {e}")
                        st.session_state.results = {} # Clear partial results on failure
        else:
            # Results exist, so don't show the "Run" button.
            pass


    def _display_step_3_results(self):
        """
        Displays the results from the experiment run, grouped by dataset and model group.
        """
        if not st.session_state.get("results"):
            return
            
        st.subheader("Experiment Results")

        for dataset_name, dataset_results in st.session_state.results.items():
            st.markdown(f"### Results for: `{dataset_name}`")
            
            # --- MODIFIED: Check for dataset-level error ---
            if dataset_results.get("error"):
                st.error(f"Error processing this dataset: {dataset_results['error']}")
                continue
            
            # --- MODIFIED: Get the 'metrics' dictionary ---
            metrics_data = dataset_results.get("metrics", {})

            if not metrics_data:
                st.warning("No results were generated for this dataset.")
                continue

            for group_name, group_results in metrics_data.items():
                st.markdown(f"#### Model Group: {group_name}")
                
                try:
                    df = pd.DataFrame.from_dict(group_results, orient="index")
                    
                    if "error" in df.columns and len(df.columns) == 1:
                        st.dataframe(df) # Show the error DataFrame
                    else:
                        st.dataframe(
                            df.style.format(
                                {
                                    "AUC": "{:.4f}",
                                    "PCC": "{:.4f}",
                                    "F1": "{:.4f}",
                                    "Recall": "{:.4f}",
                                    "BS": "{:.4f}",
                                    "KS": "{:.4f}",
                                    "PG": "{:.4f}",
                                    "H": "{:.4f}",
                                },
                                na_rep="Error" 
                            )
                        )
                except Exception as e:
                    st.error(f"Could not display results for {group_name}: {e}")
                    st.json(group_results) 


    def _calculate_benchmarks(self):
        """
        Calculates benchmark models and average AUC comparison tables.
        Populates session state with two DataFrames.
        """
        results = st.session_state.get("results", {})
        if not results:
            st.warning("No results found. Please run Step 3 first.")
            return

        # 1. Aggregate all AUC scores for each model
        model_scores: Dict[str, Dict[str, List[float]]] = {} # {group: {model: [auc1, auc2, ...]}}
        
        for dataset_name, dataset_results in results.items():
            # --- MODIFIED: Check for error and get metrics ---
            if dataset_results.get("error"):
                continue
            metrics_data = dataset_results.get("metrics", {})
            
            for group_name, group_results in metrics_data.items():
                if group_name not in model_scores:
                    model_scores[group_name] = {}
                
                for model_name, metrics in group_results.items():
                    if model_name not in model_scores[group_name]:
                        model_scores[group_name][model_name] = []
                    
                    if "AUC" in metrics and pd.notna(metrics["AUC"]): # Check for errors/NaN
                        model_scores[group_name][model_name].append(metrics["AUC"])

        # 2. Find best model AND build average AUC comparison tables
        benchmark_models: Dict[str, str] = {} # {group: 'best_model_name'}
        auc_comparison_tables: Dict[str, pd.DataFrame] = {} 
        
        for group_name, models in model_scores.items():
            avg_aucs = {}
            for model_name, auc_list in models.items():
                if auc_list: # Only consider models that ran successfully
                    avg_aucs[model_name] = np.mean(auc_list)
            
            if avg_aucs: 
                avg_auc_df = pd.DataFrame.from_dict(avg_aucs, orient='index', columns=['Average AUC'])
                avg_auc_df = avg_auc_df.sort_values(by='Average AUC', ascending=False)
                auc_comparison_tables[group_name] = avg_auc_df

                best_model = max(avg_aucs, key=avg_aucs.get)
                benchmark_models[group_name] = best_model
        
        st.session_state.benchmark_auc_comparison = auc_comparison_tables

        if not benchmark_models:
            st.error("Could not determine benchmark models. No successful runs found.")
            return

        # 3. Build the final benchmark summary table data
        final_table_data = []
        
        for dataset_name, dataset_results in results.items():
            # --- MODIFIED: Check for error and get metrics ---
            if dataset_results.get("error"):
                continue
            metrics_data = dataset_results.get("metrics", {})
            
            for group_name, best_model_name in benchmark_models.items():
                if group_name in metrics_data and best_model_name in metrics_data[group_name]:
                    metrics = metrics_data[group_name][best_model_name]
                    
                    if "error" not in metrics:
                        row = {
                            'Dataset': dataset_name,
                            'Model Group': group_name,
                            'Benchmark Model': best_model_name,
                            **metrics 
                        }
                        final_table_data.append(row)
        
        if not final_table_data:
            st.error("Failed to build benchmark table. No valid metrics found.")
            return
            
        # 4. Create and store the final summary DataFrame
        df = pd.DataFrame(final_table_data)
        all_cols = [
            'Dataset', 'Model Group', 'Benchmark Model', 
            'AUC', 'PCC', 'F1', 'Recall', 'BS', 'KS', 'PG', 'H'
        ]
        final_cols = [col for col in all_cols if col in df.columns]
        st.session_state.benchmark_results_df = df[final_cols]
        # Persist benchmark results to disk for reproducibility
        try:
            out_dir = Path(__file__).parent / "results"
            out_dir.mkdir(parents=True, exist_ok=True)
            # Use a deterministic filename so repeated runs overwrite the file
            fname = "benchmark_results.csv"
            out_path = out_dir / fname
            st.session_state.benchmark_results_df.to_csv(out_path, index=False)
            # Store the saved CSV path in session state for easy access
            st.session_state.benchmark_results_csv = str(out_path)
        except Exception as e:
            st.warning(f"Could not save benchmark results to disk: {e}")


    def _render_step_4_benchmark_analysis(self):
        st.header("Step 4: Benchmark Analysis")

        has_results = bool(st.session_state.get("results"))

        # Button only sets intent and clears old outputs
        clicked = st.button(
            "Find Benchmark Models",
            disabled=not has_results,
            key="btn_benchmark_models"
        )

        if clicked:
            st.session_state["benchmark_requested"] = True
            st.session_state["benchmark_results_df"] = None
            st.session_state["benchmark_auc_comparison"] = None
            st.session_state["run_shap"] = False

        # Compute ONLY if the user requested it
        if has_results and st.session_state.get("benchmark_requested"):
            with st.spinner("Calculating benchmark models..."):
                self._calculate_benchmarks()
            st.session_state["benchmark_requested"] = False  # consume the intent
            if st.session_state.get("benchmark_results_df") is not None:
                st.success("Benchmark analysis complete!")


    def _display_step_4_results(self):
        """
        Displays the final benchmark results.
        """
        bench_df = st.session_state.get("benchmark_results_df")

        if bench_df is not None:
            st.subheader("Benchmark Model Summary")
            st.markdown("This table shows the full performance metrics for *only* the best model from each group on each dataset.")
            df = bench_df
            st.dataframe(
                df.style.format(
                    {"AUC":"{:.4f}","PCC":"{:.4f}","F1":"{:.4f}","Recall":"{:.4f}",
                    "BS":"{:.4f}","KS":"{:.4f}","PG":"{:.4f}","H":"{:.4f}"},
                    na_rep="N/A"
                )
            )
            st.markdown("---")

            # --- Model Comparison Charts Subsection ---
            st.subheader("Model Comparison Charts")
            st.markdown("Compare benchmark metrics across models and datasets.")

            df_long = _prepare_benchmark_long(df)
            if df_long.empty:
                st.info("No benchmark metrics available for charts.")
            else:
                # Use the canonical metric list requested by the user
                metrics = ["AUC", "PCC", "F1", "Recall", "BS", "KS", "PG", "H"]
                # Build charts only for metrics present in the long dataframe
                charts = {m: make_metric_chart(df_long, m) for m in metrics}

                # Display charts in rows of up to 3 charts per row for a responsive layout
                def _chunk(seq, n):
                    for i in range(0, len(seq), n):
                        yield seq[i : i + n]

                present_metrics = [m for m in metrics if not df_long[df_long["metric"] == m].empty]
                for chunk in _chunk(present_metrics, 3):
                    cols = st.columns(len(chunk))
                    for c, met in zip(cols, chunk):
                        with c:
                            st.altair_chart(charts.get(met), use_container_width=True)

            # Continue with existing stat tests
            self._render_step_4_stat_tests()
        else:
            st.info("Run benchmark analysis to see the final summary table here.")


    def _render_step_4_stat_tests(self):
        """
        Statistical tests on paired model outputs (per dataset).
        """
        if not SCIPY_AVAILABLE:
            st.warning("SciPy is not installed. Install `scipy` to run Wilcoxon, McNemar, and DeLong tests.")
            return

        all_results = st.session_state.get("results", {})
        if not all_results:
            st.info("Run experiments in Step 3 before running statistical tests.")
            return

        # Dataset selection area (use a container instead of an expander to avoid nested expanders)
        dataset = None
        with st.container():
            st.markdown("**Paired Comparison: Select Dataset**")
            st.caption("Choose a dataset to perform paired statistical comparisons (Wilcoxon, McNemar, DeLong).")
            ds_names = list(all_results.keys())
            dataset = st.selectbox("Dataset for paired comparison:", ds_names, key="stat_tests_dataset")

        if not dataset:
            st.info("Select a dataset in the 'Paired Comparison: Select Dataset' area to enable tests.")
            return

        data_block = all_results.get(dataset, {})
        models_block = data_block.get("models", {})
        data_dict = data_block.get("data", {})
        X_test = data_dict.get("X_test"); y_test = data_dict.get("y_test")
        if X_test is None or y_test is None:
            st.info("Stored test data not available for this dataset.")
            return
        model_options = []
        for group, models in models_block.items():
            for name, model in models.items():
                if model is not None:
                    model_options.append(f"{group}::{name}")
        if len(model_options) < 2:
            st.info("Need at least two trained models to compare on this dataset.")
            return

        c1, c2 = st.columns(2)
        choice_a = c1.selectbox("Model A", model_options, key="stat_model_a")
        remaining = [m for m in model_options if m != choice_a]
        choice_b = c2.selectbox("Model B", remaining, key="stat_model_b")

        if st.button("Run statistical tests", key="btn_stat_tests"):
            try:
                grp_a, name_a = choice_a.split("::", 1)
                grp_b, name_b = choice_b.split("::", 1)
                model_a = models_block.get(grp_a, {}).get(name_a)
                model_b = models_block.get(grp_b, {}).get(name_b)
                if model_a is None or model_b is None:
                    st.error("Selected models could not be located.")
                    return

                proba_a = model_a.predict_proba(X_test)[:, 1]
                proba_b = model_b.predict_proba(X_test)[:, 1]
                pred_a = model_a.predict(X_test)
                pred_b = model_b.predict(X_test)
            except Exception as e:
                st.error(f"Failed to score selected models: {e}")
                return

            # Wilcoxon on per-sample absolute error
            wil_stat, wil_p, med_diff = _wilcoxon_abs_error_test(y_test, proba_a, proba_b)
            b, c, chi2, mc_p = _mcnemar_test(y_test, pred_a, pred_b)
            delong_res = delong_roc_test(y_test, proba_a, proba_b)

            st.markdown("##### Wilcoxon signed-rank (absolute error per sample)")
            if wil_stat is None:
                st.info("Wilcoxon test could not be computed (check data length or SciPy availability).")
            else:
                st.write({
                    "statistic": wil_stat,
                    "p_value": wil_p,
                    "median_abs_error_diff (A-B)": med_diff
                })

                # Human-readable summary
                if med_diff is not None and wil_p is not None:
                    direction = "Model A" if med_diff < 0 else ("Model B" if med_diff > 0 else "Both models")
                    verb = "had a lower" if med_diff != 0 else "had similar"
                    significance = "statistically significant" if wil_p < 0.05 else "not statistically significant"
                    st.markdown(
                        f"{direction} {verb} median absolute error than the other, and this difference was {significance} (p={wil_p:.4g})."
                    )

            st.markdown("##### McNemar's test (paired classification outcomes)")
            if chi2 is None or mc_p is None:
                reason = "no discordant pairs (models agreed on every case)" if b is not None and c is not None and (b + c) == 0 else "insufficient data or missing predictions"
                st.info(f"McNemar's test not computed: {reason}.")
            else:
                st.write({
                    "b (A correct, B wrong)": b,
                    "c (A wrong, B correct)": c,
                    "chi2": chi2,
                    "p_value": mc_p
                })
                st.caption("McNemar compares disagreements; low p_value means one model is more accurate on the discordant cases.")

            st.markdown("##### DeLong test for AUC difference")
            if delong_res is None:
                st.info("DeLong test could not be computed (need SciPy and binary labels with both classes).")
            else:
                st.write(delong_res)
                st.caption("Positive auc_diff favors Model A. Lower p_value indicates a significant AUC gap.")


    def _render_step_5_shap_analysis(self):
        """
        Renders the merged SHAP analysis with global plots and reliability checks.
        """
        st.header("Step 5: Global SHAP & Reliability Analysis")
        
        if not SHAP_AVAILABLE:
            st.error("SHAP library not found. Please install it to run this analysis: `pip install shap matplotlib`")
            return

        st.markdown("Generate SHAP summary plots (global feature importance) for the best-performing **benchmark model** from each dataset.")
        # Dataset selection for SHAP (multi-select). Defaults to all benchmarked datasets if available.
        bench_df = st.session_state.get("benchmark_results_df")
        available_ds = []
        if bench_df is not None and not bench_df.empty:
            try:
                available_ds = list(pd.unique(bench_df["Dataset"]))
            except Exception:
                available_ds = list(st.session_state.get("selected_datasets", []))
        else:
            # Fall back to uploaded datasets list if benchmarks are not yet computed
            available_ds = list(st.session_state.get("selected_datasets", []))

        # Determine default selection (do NOT assign to session_state before widget creation)
        default_sel = st.session_state.get("shap_selected_datasets", available_ds)

        # Create the multiselect widget. Streamlit will populate `st.session_state["shap_selected_datasets"]`.
        st.multiselect(
            "Datasets to run Global SHAP for (multi-select):",
            options=available_ds,
            default=default_sel,
            key="shap_selected_datasets",
            help="Choose one or more datasets to generate Global SHAP plots for. Default = all available benchmark datasets.",
        )
        
        # --- Stable SHAP toggle and settings ---
        st.markdown("**⚙️ SHAP Computation Mode**")
        st.session_state.use_stable_shap = st.checkbox(
            "Use Stable SHAP (multi-trial with rank stability)",
            value=st.session_state.get("use_stable_shap", False),
            help="Runs multiple resampled trials with stratified sampling for more robust estimates and rank stability metrics."
        )
        
        if st.session_state.use_stable_shap:
            c1, c2, c3 = st.columns(3)
            st.session_state.stable_shap_trials = c1.number_input(
                "Trials",
                min_value=1,
                max_value=50,
                value=st.session_state.get("stable_shap_trials", 1),
                help="Number of resamples for rank stability. Use 1 for a quick single resample/stability check."
            )
            st.session_state.stable_shap_bg_size = c2.number_input(
                "Background size",
                min_value=50,
                max_value=2000,
                value=st.session_state.get("stable_shap_bg_size", 50),
                step=50,
                help="Background sample size per trial."
            )
            st.session_state.stable_shap_explain_size = c3.number_input(
                "Explain size",
                min_value=50,
                max_value=2000,
                value=st.session_state.get("stable_shap_explain_size", 50),
                step=50,
                help="Test sample size per trial."
            )
            st.caption(
                f"⏱️ Estimated time: ~{st.session_state.stable_shap_trials}× slower than single-shot SHAP. "
                "Provides rank stability metrics (avg_rank, std_rank) in the global table."
            )
        else:
            st.caption("Standard single-shot SHAP (fast, no rank stability metrics).")
        
        st.markdown("---")
        st.warning("This can be slow, especially for many datasets. Plots are *not* cached.")


        if st.button("Generate Global SHAP Plots"): # Renamed button
            st.session_state.run_shap = True
        
        if st.session_state.run_shap:
            self._display_step_5_results()



    def _display_step_5_results(self):
            """
            Retrieves models and data to generate SHAP plots in two columns.
            """
            benchmark_df = st.session_state.get("benchmark_results_df")
            all_results = st.session_state.get("results", {})

            if benchmark_df is None or not all_results:
                st.error("Benchmark results are missing. Cannot run SHAP.")
                return

            # Reduce to a single best benchmark per dataset (highest AUC) to save time
            try:
                best_per_dataset_idx = benchmark_df.groupby("Dataset")["AUC"].idxmax()
                reduced_benchmark_df = benchmark_df.loc[best_per_dataset_idx].reset_index(drop=True)
            except Exception:
                reduced_benchmark_df = benchmark_df

            # Ensure we clear the run flag when this function finishes so widget changes don't re-trigger SHAP runs
            try:
                with st.spinner("Generating Global SHAP plots... This may take several minutes."):
                    # Filter reduced benchmarks by user-selected datasets (if any)
                    selected_ds = st.session_state.get("shap_selected_datasets")
                    if selected_ds:
                        try:
                            reduced_benchmark_df = reduced_benchmark_df[reduced_benchmark_df["Dataset"].isin(selected_ds)].reset_index(drop=True)
                        except Exception:
                            # If filtering fails for any reason, fall back to the unfiltered set
                            pass
    
                    if reduced_benchmark_df.empty:
                        st.warning("No benchmark entries found for the selected datasets. Adjust your selection or run Step 4 first.")
                        return
    
                    for index, row in reduced_benchmark_df.iterrows():
                        dataset = row['Dataset']
                        group = row['Model Group']
                        model_name = row['Benchmark Model']
                        
                        st.subheader(f"Global SHAP Summary: `{dataset}` (Model: `{model_name}`)")
                        
                        try:
                            # Retrieve the stored model and data from the results dictionary
                            model_data = all_results.get(dataset, {})
                            model_to_explain = model_data.get('models', {}).get(group, {}).get(model_name)
                            X_train = model_data.get('data', {}).get('X_train')
                            X_test = model_data.get('data', {}).get('X_test')
                            y_train = model_data.get('data', {}).get('y_train')  # for stratified sampling
                            y_test = model_data.get('data', {}).get('y_test')    # for stratified sampling
                            
                            if model_to_explain is None or X_train is None or X_test is None:
                                st.warning(f"Could not find stored model or data for {dataset}. Skipping.")
                                continue
                            
                            # --- SHAP COMPUTATION ---
                            
                            # Choose stable or standard SHAP based on toggle
                            use_stable = st.session_state.get("use_stable_shap", False)
                            
                            if use_stable:
                                st.caption(f"🔄 Using Stable SHAP ({st.session_state.stable_shap_trials} trials)...")
                                sv, explain_data_sample_df, shap_global_df = get_shap_values_stable(
                                    model_to_explain,
                                    X_train,
                                    X_test,
                                    y_train=y_train,
                                    y_test=y_test,
                                    n_trials=int(st.session_state.stable_shap_trials),
                                    bg_size=int(st.session_state.stable_shap_bg_size),
                                    explain_size=int(st.session_state.stable_shap_explain_size),
                                )
                            else:
                                sv, explain_data_sample_df, shap_global_df = get_shap_values(
                                    model_to_explain, X_train, X_test
                                )
                            
                            # --- Cache the global SHAP df for Step 6 ---
                            if 'global_shap_dfs' not in st.session_state:
                                st.session_state.global_shap_dfs = {}
                            st.session_state.global_shap_dfs[dataset] = shap_global_df
                            
                            # --- Display rank stability metrics if using stable SHAP ---
                            if use_stable and "avg_rank" in shap_global_df.columns and "std_rank" in shap_global_df.columns:
                                st.markdown("##### 📊 Global SHAP Table (with Rank Stability)")
                                display_cols = ["rank", "feature", "abs_mean_shap", "mean_shap", "std_shap", "avg_rank", "std_rank"]
                                display_cols = [c for c in display_cols if c in shap_global_df.columns]
                                st.dataframe(
                                    shap_global_df[display_cols].head(20).style.format({
                                        "abs_mean_shap": "{:.4f}",
                                        "mean_shap": "{:.4f}",
                                        "std_shap": "{:.4f}",
                                        "avg_rank": "{:.2f}",
                                        "std_rank": "{:.2f}",
                                    }),
                                    use_container_width=True
                                )
                                st.caption(
                                    "**avg_rank**: average rank across trials (lower = more important). "
                                    "**std_rank**: rank stability (lower = more stable)."
                                )
                            # 2. Create two columns for the plots
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                
                                st.markdown("##### Summary Plot (Bar)")
                                st.caption("Average impact (magnitude) of each feature.")
                                fig, ax = plt.subplots()
                                # Bar: explicitly pass feature names (some SHAP versions need this)
                                shap_vals = sv.values if hasattr(sv, "values") else sv
                                # Robustly coerce features to numeric floats for plotting
                                plot_df = explain_data_sample_df.copy()
                                for c in plot_df.columns:
                                    # Try numeric coercion first
                                    coerced = pd.to_numeric(plot_df[c], errors='coerce')
                                    if coerced.notna().sum() >= max(1, int(0.1 * len(plot_df))):
                                        # If at least 10% of values parse as numeric, keep numeric coercion
                                        plot_df[c] = coerced
                                    else:
                                        # Else use categorical codes (preserves distinct categories)
                                        try:
                                            plot_df[c] = pd.Categorical(plot_df[c]).codes
                                        except Exception:
                                            plot_df[c] = pd.Series(pd.Categorical(plot_df[c]).codes, index=plot_df.index)
                                    # Fill remaining NaNs with column median (or 0 if median is NaN)
                                    try:
                                        med = pd.to_numeric(plot_df[c], errors='coerce').median()
                                        if pd.isna(med):
                                            med = 0.0
                                    except Exception:
                                        med = 0.0
                                    plot_df[c] = pd.to_numeric(plot_df[c], errors='coerce').fillna(med)
    
                                # Ensure float numpy array
                                plot_array = plot_df.astype(float).values
    
                                # Ensure shap values are numeric numpy arrays (handle list/multiclass)
                                if isinstance(shap_vals, (list, tuple)):
                                    sv_list = []
                                    for part in shap_vals:
                                        part_arr = pd.DataFrame(np.asarray(part))
                                        part_arr = part_arr.apply(pd.to_numeric, errors='coerce').fillna(0)
                                        sv_list.append(part_arr.values.astype(float))
                                    shap_vals_arr = np.array(sv_list)
                                else:
                                    sv_df = pd.DataFrame(np.asarray(shap_vals))
                                    sv_df = sv_df.apply(pd.to_numeric, errors='coerce').fillna(0)
                                    shap_vals_arr = sv_df.values.astype(float)
    
                                feat_names = list(shap_global_df["feature"]) if shap_global_df is not None and "feature" in shap_global_df.columns else list(explain_data_sample_df.columns)
                                shap.summary_plot(shap_vals_arr, plot_array,
                                                  plot_type="bar",
                                                  feature_names=feat_names,
                                                  show=False)
                                st.pyplot(fig)
                                try:
                                    out_fig_dir = Path(__file__).parent / "results" / "figures"
                                    out_fig_dir.mkdir(parents=True, exist_ok=True)
                                    safe_ds = str(dataset).replace(' ', '_').replace('.csv', '')
                                    safe_model = str(model_name).replace(' ', '_').replace('/', '_')
                                    bar_path = out_fig_dir / f"shap_{safe_ds}_{safe_model}_bar.png"
                                    fig.savefig(bar_path, bbox_inches='tight', dpi=150)
                                except Exception as e_save_fig:
                                    try:
                                        st.warning(f"Could not save SHAP bar PNG: {e_save_fig}")
                                    except Exception:
                                        pass
                                plt.close(fig)
    
                            with col2:
                                st.markdown("##### Summary Plot (Dot)")
                                st.caption("Distribution of feature impacts (magnitude and direction).")
                                try:
                                    fig, ax = plt.subplots()
                                    shap_vals = sv.values if hasattr(sv, "values") else sv
    
                                    plot_df = explain_data_sample_df.copy()
                                    for c in plot_df.columns:
                                        try:
                                            if not pd.api.types.is_numeric_dtype(plot_df[c]):
                                                coerced = pd.to_numeric(plot_df[c], errors='coerce')
                                                if coerced.notna().sum() >= 1:
                                                    plot_df[c] = coerced
                                                else:
                                                    plot_df[c] = pd.Categorical(plot_df[c]).codes
                                        except Exception:
                                            plot_df[c] = pd.Categorical(plot_df[c]).codes
    
                                    try:
                                        plot_array = plot_df.astype(float).values
                                    except Exception:
                                        for c in plot_df.columns:
                                            plot_df[c] = pd.to_numeric(plot_df[c], errors='coerce').fillna(0)
                                        plot_array = plot_df.astype(float).values
    
                                    feat_names = list(shap_global_df["feature"]) if shap_global_df is not None and "feature" in shap_global_df.columns else list(explain_data_sample_df.columns)
    
                                    # Coerce shap values to numeric arrays (try robustly) and attempt plotting.
                                    try:
                                        if isinstance(shap_vals, (list, tuple)):
                                            sv_list = []
                                            for part in shap_vals:
                                                part_arr = pd.DataFrame(np.asarray(part))
                                                part_arr = part_arr.apply(pd.to_numeric, errors='coerce').fillna(0)
                                                sv_list.append(part_arr.values.astype(float))
                                            shap_vals_arr = np.array(sv_list)
                                        else:
                                            sv_df = pd.DataFrame(np.asarray(shap_vals))
                                            sv_df = sv_df.apply(pd.to_numeric, errors='coerce').fillna(0)
                                            shap_vals_arr = sv_df.values.astype(float)
    
                                        shap.summary_plot(shap_vals_arr, plot_array,
                                                          feature_names=feat_names,
                                                          show=False)
                                        st.pyplot(fig)
                                        try:
                                            out_fig_dir = Path(__file__).parent / "results" / "figures"
                                            out_fig_dir.mkdir(parents=True, exist_ok=True)
                                            safe_ds = str(dataset).replace(' ', '_').replace('.csv', '')
                                            safe_model = str(model_name).replace(' ', '_').replace('/', '_')
                                            dot_path = out_fig_dir / f"shap_{safe_ds}_{safe_model}_dot.png"
                                            fig.savefig(dot_path, bbox_inches='tight', dpi=150)
                                        except Exception as e_save_fig2:
                                            try:
                                                st.warning(f"Could not save SHAP dot PNG: {e_save_fig2}")
                                            except Exception:
                                                pass
                                    except Exception as e_plot:
                                        st.error(f"SHAP dot-plot failed: {e_plot}")
                                        try:
                                            st.write({
                                                "shap_version": getattr(shap, "__version__", "unknown"),
                                                "shap_vals_type": str(type(shap_vals)),
                                                "shap_vals_asarray_dtype": str(np.asarray(shap_vals).dtype),
                                                "shap_vals_asarray_shape": str(np.asarray(shap_vals).shape),
                                                "plot_array_dtype": str(plot_array.dtype),
                                                "plot_array_shape": str(plot_array.shape),
                                            })
                                        except Exception:
                                            pass
    
                                        try:
                                            st.caption("Sample of plot data (first 5 rows)")
                                            st.write(pd.DataFrame(plot_array).head())
                                        except Exception:
                                            pass
                                        try:
                                            st.caption("Sample of shap values (as array, first 5 rows)")
                                            sv_sample = np.asarray(shap_vals)
                                            sv_show = sv_sample[0] if sv_sample.ndim == 3 else sv_sample
                                            st.write(pd.DataFrame(sv_show).head())
                                        except Exception:
                                            pass
                                    finally:
                                        plt.close(fig)
                                except Exception as e:
                                    st.error(f"Failed to generate dot plot: {e}")
                        
                        except Exception as e:
                            st.error(f"Failed to generate SHAP plot for {dataset} - {model_name}: {e}")
            finally:
                # Reset the run flag so subsequent widget changes do not cause automatic re-runs
                try:
                    st.session_state.run_shap = False
                except Exception:
                    pass
    
    def _render_step_5_5_reliability_test(self):
        """
        Renders the UI for Step 5.5: Reliability tests for SHAP (rank stability + randomization sanity).
        Runs on-demand and persists results under `results/`.
        """
        st.header("Step 5.5: Reliability Test for SHAP (rank stability & sanity)")

        if not SHAP_AVAILABLE:
            st.error("SHAP not available. Install `shap` to run reliability tests.")
            return

        # Preset speed options with on_change callback
        def _update_reliability_preset():
            """Update trials and background size when preset changes."""
            preset = st.session_state.rel_speed_preset
            if preset.startswith("Quick"):
                st.session_state.rel_n_trials = 3
                st.session_state.rel_n_bg = 50
            elif preset.startswith("Thorough"):
                st.session_state.rel_n_trials = 30
                st.session_state.rel_n_bg = 500
            else:  # Balanced
                st.session_state.rel_n_trials = 10
                st.session_state.rel_n_bg = 200
        
        # Determine default preset based on current values
        current_trials = st.session_state.get("rel_n_trials", 3)
        current_bg = st.session_state.get("rel_n_bg", 50)
        
        if current_trials == 3 and current_bg == 50:
            default_preset = "Quick (3 trials / 50 bg)"
        elif current_trials == 30 and current_bg == 500:
            default_preset = "Thorough (30 trials / 500 bg)"
        else:
            default_preset = "Balanced (10 trials / 200 bg)"
        
        preset = st.selectbox(
            "Speed preset",
            ["Quick (3 trials / 50 bg)", "Balanced (10 trials / 200 bg)", "Thorough (30 trials / 500 bg)"],
            index=["Quick (3 trials / 50 bg)", "Balanced (10 trials / 200 bg)", "Thorough (30 trials / 500 bg)"].index(default_preset),
            key="rel_speed_preset",
            on_change=_update_reliability_preset
        )

        # Map presets to values for initial defaults
        if preset.startswith("Quick"):
            default_trials, default_bg = 3, 50
        elif preset.startswith("Thorough"):
            default_trials, default_bg = 30, 500
        else:
            default_trials, default_bg = 10, 200

        c1, c2 = st.columns(2)
        # Streamlit will populate `st.session_state` for the given keys.
        c1.number_input(
            "Trials (n)", min_value=1, max_value=200, value=int(st.session_state.get("rel_n_trials", default_trials)), key="rel_n_trials"
        )
        c2.number_input(
            "Background size", min_value=10, max_value=5000, step=10, value=int(st.session_state.get("rel_n_bg", default_bg)), key="rel_n_bg"
        )

        ds_options = list(st.session_state.get("selected_datasets", []))
        default_sel = st.session_state.get("reliability_selected_datasets", ds_options)
        st.multiselect(
            "Datasets to run reliability tests for:", options=ds_options, default=default_sel,
            key="reliability_selected_datasets",
            help="Choose datasets to run rank-stability and randomization sanity checks."
        )

        if st.button("Run Reliability Tests", key="btn_run_reliability"):
            # Capture current widget values before rerun
            # (Session state is already updated by the widgets above)
            st.session_state.run_reliability_test = True
            st.session_state.analysis_active = False  # <--- RESET Step 6 flag
            # Clear previous results
            st.session_state.reliability_results = {}
            st.session_state.reliability_ratios = {}
            st.session_state.reliability_texts = {}
            st.rerun()

        # If flagged to run or results already present, display / compute
        # Only show if explicitly requested (run_reliability_test=True) OR 
        # if results exist but not explicitly disabled (run_reliability_test is None/True, not False)
        should_display = st.session_state.get("run_reliability_test", False) or (
            st.session_state.get("reliability_results") and 
            st.session_state.get("run_reliability_test") is not False
        )
        
        if should_display:
            self._display_step_5_5_reliability_results()


    def _display_step_5_5_reliability_results(self):
        """
        Runs the reliability computations and displays results.
        Saves CSV and TXT outputs deterministically under `results/`.
        """
        selected = st.session_state.get("reliability_selected_datasets", [])
        if not selected:
            st.info("Select one or more datasets above to run reliability tests.")
            return

        all_results = st.session_state.get("results", {})
        bench_df = st.session_state.get("benchmark_results_df")
        n_trials = int(st.session_state.get("rel_n_trials", 10))
        bg_size = int(st.session_state.get("rel_n_bg", 200))

        out_dir = Path(__file__).parent / "results"
        out_dir.mkdir(parents=True, exist_ok=True)

        for ds in selected:
            with st.spinner(f"Running reliability tests for {ds}..."):
                try:
                    data_block = all_results.get(ds, {})
                    data_dict = data_block.get("data", {})
                    models_block = data_block.get("models", {})

                    # Prefer the benchmark model if available
                    chosen_group = None
                    chosen_model_name = None
                    if bench_df is not None:
                        try:
                            row = bench_df[bench_df["Dataset"] == ds]
                            if not row.empty:
                                chosen_group = row.iloc[0]["Model Group"]
                                chosen_model_name = row.iloc[0]["Benchmark Model"]
                        except Exception:
                            chosen_group = None

                    # Fallback: pick the first available trained model
                    model_to_test = None
                    if chosen_group and chosen_model_name:
                        model_to_test = models_block.get(chosen_group, {}).get(chosen_model_name)

                    if model_to_test is None:
                        # pick any model present
                        for grp, grp_models in models_block.items():
                            for nm, mm in grp_models.items():
                                if mm is not None:
                                    model_to_test = mm
                                    chosen_group = grp
                                    chosen_model_name = nm
                                    break
                            if model_to_test is not None:
                                break

                    # Attempt to obtain train/test splits
                    X_train = data_dict.get("X_train")
                    X_test = data_dict.get("X_test")
                    y_train = data_dict.get("y_train")
                    y_test = data_dict.get("y_test")

                    if model_to_test is None:
                        st.warning(f"No trained model found for dataset {ds}. Skipping.")
                        continue

                    # If required data missing, try to reconstruct from uploaded full df
                    if X_train is None or X_test is None or y_train is None:
                        if ds in st.session_state.get("full_dfs", {}):
                            try:
                                full = st.session_state.full_dfs[ds]
                                target = st.session_state.get("target_column", "target")
                                if target in full.columns:
                                    X = full.drop(columns=[target])
                                    y = full[target]
                                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique()>1 else None)
                                else:
                                    st.warning(f"Target column '{st.session_state.get('target_column')}' not found in {ds}. Skipping.")
                                    continue
                            except Exception as e:
                                st.warning(f"Failed to reconstruct train/test for {ds}: {e}")
                                continue
                        else:
                            st.warning(f"No stored data for {ds} (cannot run reliability tests). Skipping.")
                            continue

                    # Run rank stability and randomization sanity
                    try:
                        rank_df = shap_rank_stability(model_to_test, X_train, X_test, n_bg_samples=bg_size, n_trials=n_trials)
                    except Exception as e_rs:
                        st.error(f"Rank stability failed for {ds}: {e_rs}")
                        rank_df = None

                    try:
                        sanity_ratio = model_randomization_sanity(model_to_test, X_train, X_test, y_train, n_bg_samples=bg_size)
                    except Exception as e_ms:
                        st.error(f"Model randomization sanity check failed for {ds}: {e_ms}")
                        sanity_ratio = float('nan')

                    # Build textual summary
                    try:
                        summary_text = summarize_reliability(rank_df, sanity_ratio if sanity_ratio is not None else 0.0, n_trials, bg_size)
                    except Exception:
                        # Fallback summary
                        if rank_df is None or rank_df.empty:
                            summary_text = f"No rank-stability data for {ds}. Sanity ratio={sanity_ratio:.3f}"
                        else:
                            top = rank_df.sort_values('avg_rank').head(5)['feature'].tolist()
                            summary_text = f"Top features: {', '.join(map(str, top))}. Sanity ratio={sanity_ratio:.3f}."

                    # Persist outputs
                    safe_name = ds.replace(' ', '_').replace('.csv', '')
                    try:
                        if rank_df is not None:
                            csv_path = out_dir / f"reliability_table_{safe_name}.csv"
                            rank_df.to_csv(csv_path, index=False)
                            st.session_state.reliability_results[safe_name] = rank_df
                        txt_path = out_dir / f"reliability_summary_{safe_name}.txt"
                        txt_path.write_text(summary_text, encoding='utf-8')
                        st.session_state.reliability_texts[safe_name] = summary_text
                        st.session_state.reliability_ratios[safe_name] = float(sanity_ratio) if sanity_ratio is not None else float('nan')
                    except Exception as e_save:
                        st.warning(f"Could not persist reliability outputs for {ds}: {e_save}")

                    # Display
                    st.markdown(f"#### Results for `{ds}`")
                    st.write({"sanity_ratio": sanity_ratio})
                    st.markdown("**Summary:**")
                    st.text(summary_text)
                    if rank_df is not None:
                        st.markdown("**Rank-stability (top 20):**")
                        display_cols = [c for c in ["feature", "avg_rank", "std_rank"] if c in rank_df.columns]
                        st.dataframe(rank_df[display_cols].head(20).style.format({"avg_rank":"{:.2f}", "std_rank":"{:.2f}"}), use_container_width=True)

                except Exception as e_all:
                    st.error(f"Reliability pipeline failed for {ds}: {e_all}")

        # Done: reset the flag so UI changes do not auto-retrigger
        try:
            st.session_state.run_reliability_test = False
        except Exception:
            pass
    

    # --- Step 6 Render Logic ---
    def _render_step_6_local_analysis(self):
            """
            Renders the UI for the new Step 6: Local SHAP Analysis.
            """

            with st.expander("Counterfactual constraints (optional)", expanded=False):
                immutable_str = st.text_input(
                    "Immutable columns (comma-separated)", value="age,gender",
                    help="These will never be changed in counterfactual search."
                )
                lb = st.text_area("Lower bounds JSON", value="{}", help='e.g. {"age": 18}')
                ub = st.text_area("Upper bounds JSON", value="{}", help='e.g. {"ltv": 1.0}')
                
                # ✅ Store constraint UI selections so _display_step_6_results can use them
                st.session_state["immutable_str"] = immutable_str
                st.session_state["lb"] = lb
                st.session_state["ub"] = ub
                st.markdown(
                    """
                    **How to use these constraints**
                    1. *Immutable columns*: comma-separated features that must never change, e.g. `age,gender`.
                    2. *Lower bounds JSON*: enforce minimum values with JSON like `{"income": 30000, "ltv": 0.20}`.
                    3. *Upper bounds JSON*: cap values via JSON such as `{"ltv": 0.80, "debt_to_income": 0.45}`.
                    Leave any field empty/`{}` if you do not need that constraint.
                    """
                )

            st.header("Step 6: Local SHAP Analysis (Explain a Single Row)")

            if not SHAP_AVAILABLE or get_llm_explanation is None:
                st.error("SHAP or OpenAI libraries not found. Please install `shap`, `matplotlib`, `openai`, and `python-dotenv` to run this analysis.")
                return

            # --- NEW: Callback to reset analysis state ---
            def _reset_local_analysis_state():
                """Resets the flag that shows the analysis results."""
                if "analysis_active" in st.session_state:
                    st.session_state.analysis_active = False

            # 1. Select Dataset
            dataset_name = st.selectbox(
                "Select a dataset to analyze:",
                st.session_state.selected_datasets,
                index=0,
                key="local_analysis_dataset_select",
                on_change=_reset_local_analysis_state  # <--- ADDED CALLBACK
            )
            
            if not dataset_name:
                st.info("Upload a dataset in Step 1 to begin.")
                return

            try:
                # 2. Load and cache the full DataFrame
                if dataset_name not in st.session_state.full_dfs:
                    with st.spinner(f"Loading {dataset_name}..."):
                        fileobj = st.session_state.uploaded_files_map[dataset_name]
                        fileobj.seek(0)
                        st.session_state.full_dfs[dataset_name] = pd.read_csv(StringIO(fileobj.getvalue().decode("utf-8")))
                
                df = st.session_state.full_dfs[dataset_name]
                
                with st.expander("Show/Hide full data table"):
                    st.dataframe(df)
                
                # 3. Select Row
                max_idx = len(df) - 1
                row_index = st.number_input(
                    f"Select a row index (0 to {max_idx})",
                    min_value=0, max_value=max_idx, step=1,
                    key="local_analysis_row_select",
                    on_change=_reset_local_analysis_state  # <--- ADDED CALLBACK
                )
                
                # 4. Analyze Button
                if st.button("Analyze Selected Row"):
                    st.session_state.analysis_active = True  # <--- SET FLAG
                    st.session_state.run_reliability_test = False  # <--- RESET Step 5.5 flag
                    st.rerun() # Force rerun to show results
                    
                # --- NEW: Display results based on flag ---
                if st.session_state.get("analysis_active", False):
                    # Get the *current* values from state
                    current_dataset = st.session_state.local_analysis_dataset_select
                    current_row = st.session_state.local_analysis_row_select
                    
                    if current_dataset in st.session_state.full_dfs:
                        current_df = st.session_state.full_dfs[current_dataset]
                        if current_row <= (len(current_df) - 1):
                            # Call display function based on the flag
                            self._display_step_6_results(current_dataset, current_df, current_row)
                        else:
                            st.error(f"Row index {current_row} is out of bounds for {current_dataset}. Max is {len(current_df) - 1}.")
                            st.session_state.analysis_active = False # Reset flag
                    else:
                        st.error(f"Data for {current_dataset} not found. Please reload.")
                        st.session_state.analysis_active = False # Reset flag
                    
            except Exception as e:
                st.error(f"Failed to load or process dataset {dataset_name}: {e}")
                if "analysis_active" in st.session_state:
                    st.session_state.analysis_active = False # Reset on error

# --- NEW: Step 6 Display Logic ---
    def _display_step_6_results(self, dataset_name, df, row_index):
        """
        Displays the SHAP waterfall plot and LLM explanation for a single row.
        """
        import matplotlib.pyplot as plt
        benchmark_df = st.session_state.benchmark_results_df
        all_results = st.session_state.results
        target_col = st.session_state.target_column

        if benchmark_df is None:
             st.error("No benchmark models found. Please run Step 4 first.")
             return
             
        # Find the benchmark model for this specific dataset
        benchmark_row = benchmark_df[benchmark_df['Dataset'] == dataset_name]
        if benchmark_row.empty:
            st.error(f"No benchmark model found for {dataset_name}. Please run Step 4.")
            return
            
        # We take the first benchmark model found (in case of multiple groups)
        model_group = benchmark_row.iloc[0]['Model Group']
        model_name = benchmark_row.iloc[0]['Benchmark Model']
        
        st.info(f"Using benchmark model: **{model_name}** (from group: {model_group})")
        
        try:
            # Get the single row of data
            instance = df.iloc[[row_index]]
            instance_features = instance.drop(columns=[target_col])
            actual_target = instance[target_col].values[0]
            
            # Get the fitted model and training data
            model_data = all_results.get(dataset_name, {})
            model = model_data.get('models', {}).get(model_group, {}).get(model_name)
            X_train = model_data.get('data', {}).get('X_train')

            if model is None or X_train is None:
                st.error("Fitted model or training data not found. Please re-run Step 3.")
                return

            # --- Get Prediction and Explanation (in parallel) ---
            with st.spinner("Calculating local SHAP explanation..."):
                pred_proba = model.predict_proba(instance_features)[0]
                prob_class_1 = pred_proba[1] # Probability of class 1
                
                # Get the SHAP Explanation object
                explanation = get_local_shap_explanation(model, X_train, instance_features)
            
            # Display metrics
            st.subheader(f"Analysis for Row {row_index}")
            col1, col2 = st.columns(2)
            col1.metric("Actual Target", f"{actual_target}")
            col2.metric("Predicted Probability (for Class 1)", f"{prob_class_1:.4f}")
            
            st.markdown("---")
            
            # Display plots
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Waterfall Plot")
                st.caption("How each feature pushes the prediction from the base value.")
                fig, ax = plt.subplots()
                # Use max_display=12 to keep it clean (11 top features + 1 "other")
                shap.waterfall_plot(explanation, max_display=10, show=False)
                st.pyplot(fig)
                try:
                    out_fig_dir = Path(__file__).parent / "results" / "figures"
                    out_fig_dir.mkdir(parents=True, exist_ok=True)
                    safe_ds = str(dataset_name).replace(' ', '_').replace('.csv', '')
                    safe_model = str(model_name).replace(' ', '_').replace('/', '_')
                    wf_path = out_fig_dir / f"shap_{safe_ds}_{safe_model}_waterfall_row{row_index}.png"
                    fig.savefig(wf_path, bbox_inches='tight', dpi=150)
                except Exception as e_wf:
                    try:
                        st.warning(f"Could not save waterfall PNG: {e_wf}")
                    except Exception:
                        pass
                plt.close(fig)

            # --- NEW: LLM Explanation in Column 2 ---
            with col2:
                st.markdown("##### AI Generated Explanation")
                st.caption("A natural language summary of the prediction.")
                with st.spinner("Asking AI for an explanation..."):
                    commentary, error = get_llm_explanation(
                        explanation,
                        actual_target,
                        prob_class_1
                    )
                    if error:
                        st.error(f"Failed to generate explanation: {error}")
                    else:
                        st.markdown(commentary)
            
            # Store local SHAP analysis results for LaTeX report generation
            try:
                if "local_shap_analyses" not in st.session_state:
                    st.session_state.local_shap_analyses = []
                
                # Build analysis record
                analysis_record = {
                    "dataset": dataset_name,
                    "row_index": row_index,
                    "model_name": model_name,
                    "model_group": model_group,
                    "actual_target": actual_target,
                    "predicted_prob": prob_class_1,
                    "instance_features": instance_features.to_dict(orient="records")[0] if not instance_features.empty else {},
                    "ai_commentary": commentary if not error else None,
                    "waterfall_png": str(wf_path.relative_to(Path(__file__).parent)) if 'wf_path' in locals() else None,
                }
                
                # Add to list (keep only most recent 10 to avoid memory issues)
                st.session_state.local_shap_analyses.append(analysis_record)
                if len(st.session_state.local_shap_analyses) > 10:
                    st.session_state.local_shap_analyses = st.session_state.local_shap_analyses[-10:]
            except Exception:
                pass  # Non-fatal; continue without storing


            st.markdown("---")
            # --- START: COUNTERFACTUAL BLOCK ---
            st.subheader("Suggested Counterfactual (minimal, feasible change)")
            
            if st.button("Find Counterfactual", key=f"btn_find_cf_{dataset_name}_{row_index}"):
                with st.spinner("Searching for a minimal counterfactual..."):
                    
                    # 1) Prepare constraints (fast)
                    try:
                        import json
                        immut = [c.strip() for c in st.session_state.get("immutable_str", "").split(",") if c.strip()]
                    except Exception:
                        immut = []
                    constraints = CFConstraints(
                        immutable=immut,
                        lower_bounds=json.loads(st.session_state.get("lb", "{}")) if "lb" in st.session_state else {},
                        upper_bounds=json.loads(st.session_state.get("ub", "{}")) if "ub" in st.session_state else {},
                    )

                    # 2) Get global SHAP from Step 5 cache (fast)
                    global_shap_cache = st.session_state.get("global_shap_dfs", {})
                    shap_global_df = global_shap_cache.get(dataset_name)

                    if shap_global_df is None:
                        st.error("Global SHAP data not found. Please run Step 5 (Global SHAP Analysis) first to enable this feature.")
                        return # Stop the counterfactual search

                    abs_mean = None
                    try:
                        if "feature" in shap_global_df and "abs_mean" in shap_global_df:
                            abs_mean = shap_global_df.set_index("feature")["abs_mean"]
                        else:
                             st.error("Cached Global SHAP data is invalid (missing 'abs_mean'). Please re-run Step 5.")
                             return # Stop
                    except Exception as e:
                        st.error(f"Could not read cached SHAP data: {e}. Please re-run Step 5.")
                        return # Stop

                    # 3) Get direction from the current instance’s explanation (fast)
                    directions = {}
                    try:
                        # explanation.values is 1 x p
                        val = explanation.values if hasattr(explanation, "values") else None
                        if val is not None:
                            s = np.sign(val.reshape(-1))
                            directions = {f: (1 if s[i] > 0 else (-1 if s[i] < 0 else 0)) for i, f in enumerate(instance_features.columns)}
                    except Exception as e:
                        st.warning(f"Could not derive directions: {e}")
                        
                    # 4) Run the search (now much faster)
                    if abs_mean is not None and directions:
                        cf = find_counterfactual(
                            model=model,
                            x0=instance_features,
                            train_sample=X_train,
                            shap_abs_mean=abs_mean,
                            directions=directions,
                            constraints=constraints,
                            target_class=1,          # flip toward class 1 by default
                            beam_width=20,
                            max_steps=30,
                            alpha=1.0,
                            beta=0.02
                        )
                        if cf["x_cf"] is not None:
                            st.success(f"Found a counterfactual with {cf['changes']} changes (objective={cf['objective']:.3f}).")
                            st.dataframe(pd.concat([instance_features.reset_index(drop=True), cf["x_cf"]], axis=0).assign(_row=["original","counterfactual"]).set_index("_row"))
                            st.caption("Interpretation: move variables in the observed directions to flip the decision with minimal change.")
                            st.download_button("Download counterfactual CSV", data=cf["x_cf"].to_csv(index=False), file_name=f"cf_{dataset_name}_row{row_index}.csv")
                        else:
                            st.warning("No counterfactual found under current constraints and step limits. Relax bounds or increase max_steps.")
                    else:
                        st.info("Counterfactual search skipped (no SHAP abs-mean or directions).")
            # --- END: COUNTERFACTUAL BLOCK ---

            st.markdown("---")
            # --- START: NEW INTERACTIONS & PDP BLOCK ---
            st.subheader("Feature Interactions & Local Response")
            
            # 1. Put the feature selector *outside* the button
            feat = st.selectbox("Feature for PDP/ICE",
                                options=list(instance_features.columns),
                                key=f"pdp_select_{dataset_name}_{row_index}") # Unique key

            # 2. Put the analysis inside a button
            if st.button("Analyze Interactions & PDP/ICE", key=f"btn_interact_{dataset_name}_{row_index}"):
                with st.spinner("Analyzing interactions and PDP..."):
                    colA, colB = st.columns(2)
                    with colA:
                        st.markdown("**Top SHAP interactions (tree models)**")
                        try:
                            pairs = shap_top_interactions_for_tree(model, X_train.sample(min(512, len(X_train))))
                            if pairs:
                                df_pairs = pd.DataFrame(pairs, columns=["feature_i","feature_j","mean_abs_interaction"])
                                st.dataframe(df_pairs)
                            else:
                                st.caption("Not available for this model type or failed to compute.")
                        except Exception as e:
                            st.caption(f"Interaction computation failed: {e}")

                    with colB:
                        st.markdown("**ICE/PDP Plot**")
                        # Read the feature from the selectbox's state
                        selected_feat = st.session_state[f"pdp_select_{dataset_name}_{row_index}"] 
                        
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots()
                        plot_ice_pdp(ax, model, X_train, selected_feat, n_ice=100)
                        st.pyplot(fig)
                        try:
                            out_fig_dir = Path(__file__).parent / "results" / "figures"
                            out_fig_dir.mkdir(parents=True, exist_ok=True)
                            safe_ds = str(dataset_name).replace(' ', '_').replace('.csv', '')
                            safe_model = str(model_name).replace(' ', '_').replace('/', '_')
                            safe_feat = str(selected_feat).replace(' ', '_').replace('/', '_')
                            pdp_path = out_fig_dir / f"shap_{safe_ds}_{safe_model}_pdp_row{row_index}_{safe_feat}.png"
                            fig.savefig(pdp_path, bbox_inches='tight', dpi=150)
                        except Exception as e_pdp:
                            try:
                                st.warning(f"Could not save PDP/ICE PNG: {e_pdp}")
                            except Exception:
                                pass
                        plt.close(fig)
            # --- END: NEW INTERACTIONS & PDP BLOCK ---

        except Exception as e:
            st.error(f"Failed to generate local SHAP plot: {e}")
            st.exception(e) # Show full traceback
            
    def _save_model_comparison_png(self, out_dir: Path, metric: str = "AUC") -> Optional[str]:
        """
        Create and save a simple Matplotlib model-comparison PNG for the given metric
        using the `benchmark_results_df` in session state. Returns the PNG path or None.
        """
        df = st.session_state.get("benchmark_results_df")
        if df is None or df.empty:
            return None

        try:
            # pivot: index = Benchmark Model, columns = Dataset, values = metric
            pivot = df.pivot_table(index="Benchmark Model", columns="Dataset", values=metric)
        except Exception:
            try:
                pivot = df.groupby(["Benchmark Model", "Dataset"]).agg({metric: "mean"}).unstack(fill_value=np.nan)
                pivot.columns = pivot.columns.get_level_values(1)
            except Exception:
                return None

        # Use an rc context so font increases affect only this export (no global side-effects)
        with plt.rc_context({
            # Increased by ~30% from previous values
            "font.size": 21,
            "axes.titlesize": 23,
            "axes.labelsize": 21,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
        }):
            fig, ax = plt.subplots(figsize=(10, 6))
            for col in pivot.columns:
                # Remove trailing .csv from dataset labels for cleaner legends
                lbl = str(col)
                if lbl.lower().endswith('.csv'):
                    lbl = lbl[:-4]
                ax.plot(pivot.index, pivot[col], marker="o", label=lbl)

            ax.set_title(f"Model comparison ({metric})")
            ax.set_xlabel("Model")
            ax.set_ylabel(metric)
            ax.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.set_xticks(range(len(pivot.index)))
            ax.set_xticklabels([str(x) for x in pivot.index], rotation=45, ha="right")
            fig.tight_layout()

        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"model_comparison_{metric}.png"
            fig.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return str(out_path)
        except Exception:
            plt.close(fig)
            return None

    def _summarise_models_from_code(self, df: pd.DataFrame, client) -> str:
        """
        Inspect `models.MODELS` and the benchmark DataFrame to build a small
        code-context text block, then ask the LLM to summarise model families.
        Returns the plain-text LLM response or an empty string on failure.
        """
        try:
            import models as models_module
            from models import MODELS as MODELS_DICT
        except Exception as e:
            st.warning(f"Could not import models.py for model summarization: {e}")
            return ""

        # Collect distinct model groups from the benchmark DF
        groups = []
        try:
            if "Model Group" in df.columns:
                groups = [str(x) for x in pd.unique(df["Model Group"].dropna())]
        except Exception:
            groups = []

        # Collect top benchmark models by AUC (up to ~8 unique names)
        top_models = []
        try:
            if "Benchmark Model" in df.columns and "AUC" in df.columns:
                tmp = df[["Benchmark Model", "AUC"]].dropna()
                tmp = tmp.sort_values("AUC", ascending=False)
                for name in tmp["Benchmark Model"].astype(str).tolist():
                    if name not in top_models:
                        top_models.append(name)
                    if len(top_models) >= 8:
                        break
            elif "Benchmark Model" in df.columns:
                top_models = list(pd.unique(df["Benchmark Model"].astype(str)))[:8]
        except Exception:
            top_models = []

        # Find builder functions in MODELS_DICT that match the top model names
        snippets = []
        try:
            for g in (groups or list(MODELS_DICT.keys())):
                group_dict = MODELS_DICT.get(g, {})
                for model_name, builder in group_dict.items():
                    if model_name in top_models:
                        try:
                            src = inspect.getsource(builder)
                        except Exception:
                            # Fallback: try to get __call__ source or skip
                            try:
                                src = inspect.getsource(builder.__call__)
                            except Exception:
                                src = f"# source not available for {g}/{model_name}\n"
                        snippets.append(f"### {g}/{model_name}\n{src}")

            # If no snippets found using groups, try a global search across MODELS_DICT
            if not snippets:
                for g, group_dict in MODELS_DICT.items():
                    for model_name, builder in group_dict.items():
                        if model_name in top_models:
                            try:
                                src = inspect.getsource(builder)
                            except Exception:
                                src = f"# source not available for {g}/{model_name}\n"
                            snippets.append(f"### {g}/{model_name}\n{src}")
        except Exception as e:
            st.warning(f"Error while extracting builder source: {e}")

        snippet_text = "\n\n".join(snippets)

        # Prepare LLM call
        try:
            system_msg = (
                "You are an assistant that writes concise, accurate descriptions of machine-learning model families for academic papers."
            )
            user_msg = (
                "The following code snippets are builder functions that construct models used in a credit-risk benchmarking study. "
                "Group these builders into algorithm families (e.g., logistic regression, regularised logistic regression, random forests, "
                "bagged CART, AdaBoost, gradient boosting, KNN, neural nets, XGBoost, LightGBM, etc.). "
                "For each family, produce a single bullet that names the family, lists the internal labels (Model Group / Benchmark Model), "
                "and provides a 1–3 sentence academic-style description of the family's behavior and suitability for credit-risk modelling.\n\n"
            )
            if snippet_text:
                user_msg = user_msg + "CODE SNIPPETS:\n\n" + snippet_text
            else:
                user_msg = user_msg + "(No code snippets were found or accessible.)"

            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]

            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=600,
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            st.warning(f"Model-family summarization failed: {e}")
            return ""

    def _generate_latex_from_ai(self, csv_path: str, png_path: str, api_key: str) -> Optional[str]:
        """
        Use the provided API key to call OpenAI and produce a LaTeX
        'Results' section that includes the benchmark table and the AUC PNG.
        Returns the path to the generated .tex file or None on failure.
        """
        

    def _ai_summarise_shap_reliability(self, api_key: str, dataset: str, stab_df: Optional[pd.DataFrame], sanity_ratio: Optional[float], n_trials: int, bg_size: int) -> Optional[str]:
        """
        Use OpenAI to generate a short (2-4 sentence) concise summary of Global SHAP
        findings and an explanation of the Reliability Analysis (rank stability & sanity ratio).
        Returns the generated plain-text string or None on failure.
        """
        if openai is None:
            st.warning("OpenAI client library not available. Install the 'openai' package.")
            return None

        if not api_key:
            st.warning("No OpenAI API key provided.")
            return None

        # Prepare a small structured prompt with top features and reliability metrics
        try:
            # Build top-features text
            top_text = ""
            if stab_df is not None and not stab_df.empty:
                try:
                    # Prefer columns: feature, abs_mean_shap, avg_rank, std_rank
                    cols = stab_df.columns.tolist()
                    rows = stab_df.head(8)
                    lines = []
                    for _, r in rows.iterrows():
                        feat = str(r.get("feature", "<feature>"))
                        abs_mean = r.get("abs_mean", r.get("abs_mean_shap", r.get("abs_mean_shap", None)))
                        avg_rank = r.get("avg_rank", None)
                        std_rank = r.get("std_rank", None)
                        parts = [f"{feat}"]
                        if pd.notna(abs_mean):
                            parts.append(f"abs_mean={float(abs_mean):.4f}")
                        if pd.notna(avg_rank):
                            parts.append(f"avg_rank={float(avg_rank):.2f}")
                        if pd.notna(std_rank):
                            parts.append(f"std_rank={float(std_rank):.2f}")
                        lines.append(" (".join(parts) + ")" if False else ", ".join(parts))
                    top_text = "\n".join(lines)
                except Exception:
                    top_text = "(Top features unavailable)"
            else:
                top_text = "(No SHAP rank table available)"

            ratio_text = f"Sanity ratio: {sanity_ratio:.3f}" if sanity_ratio is not None and pd.notna(sanity_ratio) else "Sanity ratio: N/A"

            prompt = (
                "You are an assistant that writes concise, scientific summaries for machine-learning explainability outputs. "
                "Given Global SHAP rank/stability information and a reliability metric, produce a short 2-4 sentence paragraph that: (1) summarises the most important features and their relative influence, and (2) explains the reliability analysis (what the sanity ratio means and how avg_rank/std_rank reflect stability) and gives one actionable recommendation (e.g., increase trials, increase background size, or treat results cautiously). "
                "Keep language non-technical but precise and suitable for a figure caption or brief report blurb."
                "\n\nDATA:\n"
                f"Dataset: {dataset}\n"
                f"Top features (one per line):\n{top_text}\n"
                f"{ratio_text}\n"
                f"Stable-SHAP trials: {n_trials}, background size: {bg_size}\n"
            )

            client = openai.OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an assistant that writes concise scientific summaries."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=180,
                temperature=0.0,
            )
            text = resp.choices[0].message.content.strip()
            return text
        except Exception as e:
            st.warning(f"AI summarisation failed: {e}")
            return None

    def _generate_latex_from_ai(self, csv_path: str, png_path: str, api_key: str) -> Optional[str]:
        """
        Use the provided API key to call OpenAI and produce a LaTeX
        'Results' section that includes the benchmark table and the AUC PNG.
        Returns the path to the generated .tex file or None on failure.
        """
        if openai is None:
            st.warning("The 'openai' package is not installed. Install it to enable AI report generation.")
            return None

        # reading CSV for LaTeX generation
        try:
            df = pd.read_csv(csv_path)
            # CSV loaded successfully
        except Exception as e:
            st.warning(f"Could not read CSV for AI report generation: {e}")
            return None

        # Convert table to LaTeX string
        # converting table to LaTeX
        try:
            # Create a copy of the dataframe and escape underscores in string columns
            df_latex = df.copy()
            for col in df_latex.columns:
                if df_latex[col].dtype == 'object':  # string columns
                    df_latex[col] = df_latex[col].astype(str).str.replace('_', '\\_', regex=False)

            table_tex = df_latex.to_latex(index=False, caption="Benchmark Results", label="tab:benchmark", float_format="{:.4f}".format, escape=False)
        except Exception:
            # Fallback: simple tabular conversion with manual escaping
            df_latex = df.copy()
            for col in df_latex.columns:
                if df_latex[col].dtype == 'object':
                    df_latex[col] = df_latex[col].astype(str).str.replace('_', '\\_', regex=False)
            table_tex = df_latex.to_latex(index=False, escape=False)

        # Wrap the tabular environment with \resizebox{\textwidth}{!}{% ... }
        try:
            # Find the tabular block and wrap it so captions/labels remain outside
            tabular_match = re.search(r"(\\begin\{tabular\}.*?\\end\{tabular\})", table_tex, flags=re.DOTALL)
            if tabular_match:
                tabular_block = tabular_match.group(1)
                wrapped = "\\resizebox{\\textwidth}{!}{%\n" + tabular_block + "\n}"
                table_tex = table_tex.replace(tabular_block, wrapped)
        except Exception:
            # If wrapping fails, continue with unwrapped table_tex
            pass

        # Build LaTeX skeleton
        # Use a LaTeX-relative figures/ path for inclusion
        figure_basename = os.path.basename(png_path) if png_path else "model_comparison_AUC.png"
        # Use POSIX-style forward slash in LaTeX path regardless of OS
        figure_filename = f"figures/{figure_basename}"
        # figure filename for LaTeX: {figure_filename}

        # Default single-figure block (well-indented)
        figure_block = rf"""
\begin{{figure}}[ht]
    \centering
    \includegraphics[width=0.9\linewidth]{{{figure_filename}}}
    \caption{{Model comparison (AUC).}}
    \label{{fig:auc}}
\end{{figure}}
"""

        # Detect all model_comparison PNGs and build a 2x4 grid figure block if possible
        try:
            # Prefer figures folder next to the CSV (e.g., results/figures)
            csv_parent = Path(csv_path).parent
            candidate_dirs = [csv_parent / "figures", Path(__file__).parent / "results" / "figures"]
            figures_dir = None
            for d in candidate_dirs:
                if d.exists():
                    figures_dir = d
                    break
            if figures_dir is None:
                figures_dir = candidate_dirs[0]

            # Desired metric order for consistent layout
            # Exclude PCC and PG from the image grid (they remain present in the CSV/table)
            metrics_order = ["AUC", "F1", "Recall", "BS", "KS", "H"]
            imgs = []
            for m in metrics_order:
                fname = f"model_comparison_{m}.png"
                p = figures_dir / fname
                if p.exists():
                    imgs.append(f"figures/{fname}")

            # Fallback: if no images found by the ordered list, try globbing any model_comparison_*.png
            if not imgs:
                for p in sorted(figures_dir.glob("model_comparison_*.png")):
                    imgs.append(f"figures/{p.name}")

            # Ensure we have up to 6 slots (2 rows × 3 cols); pad with empty strings if necessary
            imgs = imgs[:6]
            while len(imgs) < 6:
                imgs.append("")

            # Build 2x3 tabular latex block (2 rows, 3 columns)
            row1 = imgs[0:3]
            row2 = imgs[3:6]

            def img_tex(path):
                if not path:
                    return ""
                return f"\\includegraphics[width=0.32\\textwidth]{{{path}}}"

            def build_row_tex(row):
                # create tex for non-empty cells; return None for an all-empty row
                cells = [img_tex(p) for p in row if p]
                if not cells:
                    return None
                return " &\n    ".join(cells)

            row_texts = [build_row_tex(r) for r in (row1, row2)]
            # Keep only non-empty rows to avoid stray blank rows like " &\n &\n"
            included_rows = [r for r in row_texts if r]

            if included_rows:
                rows_joined = " \\\\\n    ".join(included_rows)
                # overwrite figure_block with multi-image layout (2x3) using only included rows
                figure_block = rf"""
\begin{{figure}}[ht]
    \centering
    \begin{{tabular}}{{ccc}}
        {rows_joined}
    \end{{tabular}}
    \caption{{Model comparison across metrics.}}
    \label{{fig:model_comparisons}}
    \label{{fig:auc}}
\end{{figure}}
"""
            else:
                # No images found; keep the default single-image figure_block
                pass
        except Exception as e:
            st.warning(f"Could not build multi-image figure block: {e}")
            # keep default single-figure 'figure_block' defined above

        # Two-phase LLM pipeline: (1) summarise model families from code, (2) generate LaTeX Results
        try:
            client = openai.OpenAI(api_key=api_key)
        except Exception as e:
            st.warning(f"Could not initialise OpenAI client: {e}")
            return None

        # Phase 1: summarise model families from models.py (best-effort)
        try:
            model_context = self._summarise_models_from_code(df, client)
        except Exception as e:
            st.warning(f"Model summarisation step failed: {e}")
            model_context = ""

        # Phase 2: build the Results-section prompt that uses the model-context
        try:
            user_instructions = (
                "Produce a LaTeX-formatted \\section{Results} of about 180-250 words. "
                "Start with an overview of overall performance across model families. "
                "Summarise key findings from the benchmark table, focusing on AUC as the primary metric and commenting on F1/Recall differences when relevant. "
                "When first mentioning each major model family (logistic regression, regularised logistic regression, random forests, bagged CART, AdaBoost / Boost-DT, gradient boosting, KNN, neural nets, XGBoost, LightGBM, etc.), "
                "use the MODEL-FAMILY DESCRIPTIONS to add a short clause (e.g. 'random forests (ensembles of bagged decision trees)'). "
                "Reference the figure strictly as \\ref{fig:auc}. End with implications for credit-risk modelling (which families are most promising and why), consistent with the numerical results. "
                "Use proper LaTeX formatting and escape underscores as \\_ in all model names (e.g. lr_reg -> lr\\_reg)."
            )

            model_context_block = model_context if model_context else "(No model-family context was available.)"

            user_msg = (
                user_instructions
                + "\n\nBENCHMARK TABLE (CSV format):\n"
                + df.to_string()
                + "\n\nMODEL-FAMILY DESCRIPTIONS:\n"
                + model_context_block
                + "\n\nNote: the main AUC figure is labelled 'fig:auc'."
            )

            messages = [
                {"role": "system", "content": "You are an assistant that writes LaTeX for academic papers. Be concise, numerically faithful to the data, and use proper LaTeX commands."},
                {"role": "user", "content": user_msg},
            ]

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=800,
                temperature=0.2,
            )
            ai_text = response.choices[0].message.content
        except Exception as e:
            st.warning(f"AI generation failed: {e}")
            st.exception(e)
            return None

        # --- NEW: optionally generate per-dataset AI SHAP reliability summaries
        dataset_summaries_tex = ""
        try:
            # Collect datasets from global_shap_dfs (these are the ones that actually had SHAP run)
            datasets = []
            try:
                shap_cache = st.session_state.get("global_shap_dfs", {})
                if shap_cache:
                    datasets = list(shap_cache.keys())
                # Fallback to benchmark CSV if no SHAP cache
                elif "Dataset" in df.columns:
                    datasets = list(pd.unique(df["Dataset"]))
            except Exception:
                datasets = []

            # Limit to a reasonable number to avoid excessive API calls
            MAX_SUMMARIES = 6
            summary_calls = 0

            def _latex_escape(s: str) -> str:
                if s is None:
                    return ""
                esc = str(s)
                # minimal escaping for LaTeX special chars likely to appear
                for a, b in [("\\", "\\textbackslash{}"), ("$", "\\$"), ("%", "\\%"), ("_", "\\_"), ("&", "\\&"), ("#", "\\#"), ("{", "\\{"), ("}", "\\}")]:
                    esc = esc.replace(a, b)
                return esc

            if datasets and api_key:
                for ds_name in datasets:
                    if summary_calls >= MAX_SUMMARIES:
                        break
                    # Try to use cached reliability results if available; else use global_shap_dfs
                    stab_df = None
                    sanity_ratio = None
                    try:
                        stab_df = st.session_state.get("reliability_results", {}).get(ds_name)
                    except Exception:
                        stab_df = None
                    try:
                        sanity_ratio = st.session_state.get("reliability_ratios", {}).get(ds_name)
                    except Exception:
                        sanity_ratio = None

                    # If no stab_df, try global_shap_dfs (may lack ranks)
                    if stab_df is None:
                        try:
                            shap_cache = st.session_state.get("global_shap_dfs", {})
                            shap_df = shap_cache.get(ds_name)
                            # Provide a small fallback table (top features by abs_mean if available)
                            if shap_df is not None and not shap_df.empty:
                                stab_df = shap_df.head(8)
                        except Exception:
                            shap_df = None

                    try:
                        summary_calls += 1
                        ai_sum = self._ai_summarise_shap_reliability(api_key, ds_name, stab_df, sanity_ratio, int(n_trials) if 'n_trials' in locals() else 10, int(n_bg) if 'n_bg' in locals() else 200)
                    except Exception:
                        ai_sum = None

                    if ai_sum:
                        ds_label = _latex_escape(ds_name)
                        ai_escaped = _latex_escape(ai_sum)
                        dataset_summaries_tex += f"\\subsubsection*{{Global SHAP — {ds_label}}}\n{ai_escaped}\n\n"
                        
                        # Add bar and dot images in two columns
                        try:
                            bench_df = st.session_state.get("benchmark_results_df")
                            if bench_df is not None and not bench_df.empty:
                                # Find the benchmark model for this dataset
                                ds_rows = bench_df[bench_df["Dataset"] == ds_name]
                                if not ds_rows.empty:
                                    model_name = ds_rows.iloc[0]["Benchmark Model"]
                                    safe_ds = str(ds_name).replace(' ', '_').replace('.csv', '')
                                    safe_model = str(model_name).replace(' ', '_').replace('/', '_')
                                    
                                    bar_fig = f"figures/shap_{safe_ds}_{safe_model}_bar.png"
                                    dot_fig = f"figures/shap_{safe_ds}_{safe_model}_dot.png"
                                    
                                    # Check if files exist
                                    bar_path = Path(__file__).parent / "results" / bar_fig
                                    dot_path = Path(__file__).parent / "results" / dot_fig
                                    
                                    # Always add figures regardless of file existence (LaTeX will handle missing files)
                                    dataset_summaries_tex += "\\begin{figure}[H]\n"
                                    dataset_summaries_tex += "\\centering\n"
                                    dataset_summaries_tex += "\\begin{tabular}{cc}\n"
                                    dataset_summaries_tex += f"\\includegraphics[width=0.48\\textwidth]{{{bar_fig}}} &\n"
                                    dataset_summaries_tex += f"\\includegraphics[width=0.48\\textwidth]{{{dot_fig}}} \\\\\n"
                                    dataset_summaries_tex += "Bar Plot & Summary Plot (Dot) \\\\\n"
                                    dataset_summaries_tex += "\\end{tabular}\n"
                                    dataset_summaries_tex += f"\\caption{{Global SHAP plots for {ds_label} using {_latex_escape(model_name)}}}\n"
                                    dataset_summaries_tex += f"\\label{{fig:shap_{safe_ds}_{safe_model}}}\n"
                                    dataset_summaries_tex += "\\end{figure}\n\n"
                        except Exception as e:
                            # Non-fatal; continue without figures
                            st.warning(f"Could not add SHAP images for {ds_name}: {e}")
        except Exception:
            # Non-fatal; continue without per-dataset summaries
            dataset_summaries_tex = ""

        # Define a single LaTeX escape helper function to use throughout
        def _latex_escape(s: str) -> str:
            if s is None:
                return ""
            esc = str(s)
            # Important: escape backslash first, then other special chars
            for a, b in [("\\", "\\textbackslash "), ("$", "\\$"), ("%", "\\%"), ("_", "\\_"), ("&", "\\&"), ("#", "\\#"), ("{", "\\{"), ("}", "\\}")]:
                esc = esc.replace(a, b)
            return esc

        # If no per-dataset summaries were generated, attempt a single global summary fallback
        try:
            if (not dataset_summaries_tex.strip()) and api_key:
                # Compute an average sanity ratio if any are available
                ratios = []
                try:
                    ratios = [v for v in (st.session_state.get("reliability_ratios") or {}).values() if v is not None]
                except Exception:
                    ratios = []
                avg_ratio = float(np.mean(ratios)) if ratios else None

                overall_ai = None
                try:
                    overall_ai = self._ai_summarise_shap_reliability(api_key, "Overall", None, avg_ratio, int(st.session_state.get("rel_n_trials", 10)), int(st.session_state.get("rel_n_bg", 200)))
                except Exception:
                    overall_ai = None

                if overall_ai:
                    dataset_summaries_tex = f"\\subsection*{{SHAP Reliability Summary}}\\n{_latex_escape(overall_ai)}\\n\\n"
        except Exception:
            # safe fallback: leave empty
            pass

        # --- NEW: Build Local SHAP Analysis subsection if available ---
        local_shap_tex = ""
        try:
            analyses = st.session_state.get("local_shap_analyses", [])
            if analyses:
                # Only include the most recent analysis (single example)
                rec = analyses[-1]
                
                ds_esc = _latex_escape(rec.get("dataset", "Unknown"))
                model_esc = _latex_escape(rec.get("model_name", "Unknown"))
                row_idx = rec.get("row_index", "?")
                actual = rec.get("actual_target", "?")
                pred_prob = rec.get("predicted_prob", 0.0)
                commentary = rec.get("ai_commentary", "")
                waterfall_png = rec.get("waterfall_png", None)
                
                local_shap_tex = "\\subsection*{Local SHAP Analysis}\n\n"
                local_shap_tex += f"\\subsubsection*{{Analysis for Row {row_idx}}}\n"
                local_shap_tex += f"\\textbf{{Dataset:}} {ds_esc} \\\\\n"
                local_shap_tex += f"\\textbf{{Model:}} {model_esc} \\\\\n"
                local_shap_tex += f"\\textbf{{Actual Target:}} {actual} \\\\\n"
                local_shap_tex += f"\\textbf{{Predicted Probability (for Class 1):}} {pred_prob:.4f}\n\n"
                
                # Create two-column layout: Waterfall Plot on left, AI Explanation on right
                if waterfall_png and commentary:
                    # Use forward slashes for LaTeX paths and strip results/ prefix
                    png_path = waterfall_png.replace("\\", "/")
                    # Remove results/ prefix if present for LaTeX relative path
                    if png_path.startswith("results/"):
                        png_path = png_path[8:]  # Remove "results/"
                    
                    commentary_esc = _latex_escape(commentary)
                    
                    local_shap_tex += "\\begin{figure}[H]\n"
                    local_shap_tex += "\\centering\n"
                    local_shap_tex += "\\begin{tabular}{p{0.48\\textwidth}p{0.48\\textwidth}}\n"
                    local_shap_tex += "\\textbf{Waterfall Plot} & \\textbf{AI Generated Explanation} \\\\\n"
                    local_shap_tex += "\\footnotesize How each feature pushes the prediction from the base value. & \\footnotesize A natural language summary of the prediction. \\\\\n"
                    local_shap_tex += f"\\includegraphics[width=0.48\\textwidth]{{{png_path}}} &\n"
                    local_shap_tex += f"\\begin{{minipage}}[t]{{0.48\\textwidth}}\n"
                    local_shap_tex += f"\\vspace{{0pt}}\n"
                    local_shap_tex += f"\\small {commentary_esc}\n"
                    local_shap_tex += f"\\end{{minipage}}\n"
                    local_shap_tex += "\\\\\n"
                    local_shap_tex += "\\end{tabular}\n"
                    local_shap_tex += f"\\caption{{Local SHAP analysis for {ds_esc}, row {row_idx}: Waterfall plot and AI-generated explanation.}}\n"
                    local_shap_tex += f"\\label{{fig:local_shap_row{row_idx}}}\n"
                    local_shap_tex += "\\end{figure}\n\n"
                elif waterfall_png:
                    # Only waterfall available
                    png_path = waterfall_png.replace("\\", "/")
                    if png_path.startswith("results/"):
                        png_path = png_path[8:]
                    local_shap_tex += f"""\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.75\\textwidth]{{{png_path}}}
    \\caption{{Waterfall plot for {ds_esc}, row {row_idx}.}}
    \\label{{fig:waterfall_row{row_idx}}}
\\end{{figure}}

"""
                elif commentary:
                    # Only commentary available
                    commentary_esc = _latex_escape(commentary)
                    local_shap_tex += f"\\textbf{{AI Generated Explanation:}}\n\n{commentary_esc}\n\n"
        except Exception as e:
            # Non-fatal; continue without local SHAP section
            local_shap_tex = ""
        
        # Compose final LaTeX: AI text + per-dataset summaries + local SHAP examples + table + figure
        final_tex = ai_text + "\n\n" + dataset_summaries_tex + local_shap_tex + "\n% Benchmark Results Table:\n" + table_tex + "\n" + figure_block

        # Save to report directory (place .tex in the same 'results' folder as the CSV)
        # saving LaTeX file
        try:
            report_dir = Path(csv_path).parent
            report_dir.mkdir(parents=True, exist_ok=True)
            out_tex = report_dir / "results_section.tex"
            # saving to path
            with open(out_tex, "w", encoding="utf-8") as f:
                f.write(final_tex)
            # file saved successfully
            return str(out_tex)
        except Exception as e:
            st.warning(f"Could not save LaTeX file: {e}")
            st.exception(e)
            return None
            
    def run(self):
        """
        Run the main application logic and render the UI.
        """
        # One-click helper to expand all sections (handy before printing/exporting)
        # If clicked: ensure benchmarks exist (compute if needed) and persist CSV, then
        # the expand-preview script queued by the helper will run at the end of `run()`.
        try:
            clicked_generate = render_generate_report_button()
        except Exception:
            clicked_generate = False

        if clicked_generate:
            # Ensure benchmark results exist: compute if missing
            if st.session_state.get("benchmark_results_df") is None:
                with st.spinner("Computing benchmark models before export..."):
                    try:
                        self._calculate_benchmarks()
                    except Exception as e:
                        st.error(f"Failed to compute benchmarks for export: {e}")

            # If benchmarks are present now, (re)write deterministic CSV so Generate Report
            # always produces an up-to-date `results/benchmark_results.csv`.
            if st.session_state.get("benchmark_results_df") is not None:
                try:
                    out_dir = Path(__file__).parent / "results"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / "benchmark_results.csv"
                    st.session_state.benchmark_results_df.to_csv(out_path, index=False)
                    st.session_state.benchmark_results_csv = str(out_path)
                    st.success(f"Benchmark CSV saved: {out_path}")
                except Exception as e:
                    st.warning(f"Could not save benchmark CSV for export: {e}")
                # Also save model comparison charts as PNGs into a `report/` folder
                try:
                    report_dir = Path(__file__).parent / "report"
                    report_dir.mkdir(parents=True, exist_ok=True)

                    # Prepare long-form dataframe for charts
                    try:
                        df_long = _prepare_benchmark_long(st.session_state.benchmark_results_df)
                    except Exception:
                        df_long = None

                    if df_long is not None and not df_long.empty:
                        # Fixed model order consistent with Altair charts
                        model_order = ["lr","lr_reg","adaboost","Bag-CART","BagNN","Boost-DT","RF","SGB","KNN","XGB","LGBM","DL"]
                        metrics = ["AUC", "PCC", "F1", "Recall", "BS", "KS", "PG", "H"]
                        present_metrics = [m for m in metrics if not df_long[df_long["metric"] == m].empty]

                        for met in present_metrics:
                            try:
                                d = df_long[df_long["metric"] == met].copy()
                                # pivot to have models on x and datasets as series
                                pivot = d.pivot(index="model", columns="dataset", values="value")
                                # Reindex to ensure consistent model order
                                pivot = pivot.reindex(index=model_order)

                                import matplotlib.pyplot as _plt
                                # Use a local rc_context for slightly larger export fonts
                                with _plt.rc_context({
                                    # Increased by ~30% from previous values
                                    "font.size": 18,
                                    "axes.titlesize": 21,
                                    "axes.labelsize": 18,
                                    "xtick.labelsize": 15,
                                    "ytick.labelsize": 15,
                                    "legend.fontsize": 15,
                                }):
                                    fig, ax = _plt.subplots(figsize=(10, 4))
                                    for col in pivot.columns:
                                        # Remove trailing .csv from dataset labels for cleaner legends
                                        lbl = str(col)
                                        if lbl.lower().endswith('.csv'):
                                            lbl = lbl[:-4]
                                        ax.plot(pivot.index, pivot[col], marker="o", label=lbl)

                                    ax.set_title(f"Model Comparison: {met}")
                                    ax.set_xlabel("Model")
                                    ax.set_ylabel(met)
                                    ax.tick_params(axis='x', rotation=45)
                                    ax.legend(loc='best')
                                    _plt.tight_layout()
                                    png_name = f"model_comparison_{met.replace(' ', '_')}.png"
                                    png_path = report_dir / png_name
                                    fig.savefig(png_path, dpi=200)
                                    _plt.close(fig)
                                # Also copy the PNG into `results/figures` so LaTeX can reference `figures/...`
                                try:
                                    results_fig_dir = Path(__file__).parent / "results" / "figures"
                                    results_fig_dir.mkdir(parents=True, exist_ok=True)
                                    shutil.copy(str(png_path), results_fig_dir / png_name)
                                except Exception:
                                    pass
                            except Exception as e:
                                st.warning(f"Could not save chart for metric '{met}': {e}")
                        st.success(f"Saved model comparison PNGs to: {report_dir}")
                    else:
                        st.info("No benchmark metrics available to save charts.")
                except Exception as e:
                    st.warning(f"Could not save model comparison charts: {e}")
        
        # AI LaTeX generation section (standalone, outside Generate Report)
        st.markdown("---")
        st.subheader("Generate LaTeX Results Section")
        
        # Automatically get API key from .env file
        api_key = os.getenv("OPENAI_API_KEY")
        
        if api_key:
            st.info("✅ OpenAI API key loaded from .env file")
            
            # Check if benchmark CSV exists
            csv_path = st.session_state.get("benchmark_results_csv")
            has_csv = csv_path and Path(csv_path).exists()
            
            if not has_csv:
                st.warning("⚠️ Benchmark CSV not found. Click 'Generate Report' button above first to create the required files.")
            
            if st.button("Generate LaTeX Results Section", key="generate_latex", disabled=not has_csv):
                report_dir = Path(__file__).parent / "report"
                results_fig_dir = Path(__file__).parent / "results" / "figures"
                png_path = results_fig_dir / "model_comparison_AUC.png"
                
                # CSV and PNG paths (not shown to user)
                
                if csv_path and Path(csv_path).exists():
                    with st.spinner("Generating LaTeX results section with AI..."):
                        try:
                            latex_path = self._generate_latex_from_ai(csv_path, str(png_path), api_key)
                            # LaTeX generation returned (path stored in latex_path)
                        except Exception as e:
                            st.error(f"Exception during LaTeX generation: {e}")
                            st.exception(e)
                            latex_path = None
                        
                    if latex_path and Path(latex_path).exists():
                        st.success(f"✅ LaTeX results section successfully generated and saved to: {latex_path}")
                        
                        # Show download button
                        try:
                            with open(latex_path, "r", encoding="utf-8") as f:
                                latex_content = f.read()

                            st.download_button(
                                label="Download LaTeX Results Section",
                                data=latex_content,
                                file_name="results_section.tex",
                                mime="text/plain",
                            )

                            # Show preview with a copy-to-clipboard button
                            with st.expander("Preview LaTeX Content", expanded=False):
                                st.code(latex_content, language="latex")
                                try:
                                    import json
                                    safe_text = json.dumps(latex_content)
                                    html = f"""
<div>
  <button id="copy-latex-btn">Copy LaTeX to clipboard</button>
</div>
<script>
  const txt = {safe_text};
  const btn = document.getElementById('copy-latex-btn');
  btn.addEventListener('click', async () => {{
    try {{
      await navigator.clipboard.writeText(txt);
      btn.innerText = 'Copied!';
      setTimeout(() => btn.innerText = 'Copy LaTeX to clipboard', 2000);
    }} catch (e) {{
      alert('Copy failed: ' + e);
    }}
  }});
</script>
"""
                                    st.components.v1.html(html, height=80)
                                except Exception:
                                    # If clipboard API not available, silently continue
                                    pass
                        except Exception as e:
                            st.error(f"Error reading generated LaTeX file: {e}")
                    else:
                        st.error("❌ Failed to generate LaTeX results section. Check the debug information above.")
                else:
                    st.error("❌ Benchmark CSV not found. Please ensure benchmarks are computed first by clicking 'Generate Report' button above.")
        else:
            st.error("❌ OpenAI API key not found in .env file. Please add OPENAI_API_KEY to your .env file.")
        
        # --- Step 1: Datasets ---
        self._render_step_1_dataset_selection()
        self._display_step_1_results()
        self._render_step_1_3_ydata_profiles()
        st.markdown("---")

        # --- NEW: Step 1.25 — Paper-Style Feature Importance ---
        # --- Step 1.25 — Paper-Style Feature Importance (RF & L1-LR) ---
        with st.expander("Step 1.25: Paper-Style Feature Importance (RF & L1-LR)", expanded=False):
            have_data = bool(st.session_state.get("selected_datasets")) and bool(st.session_state.get("uploaded_files_map"))
            target = st.session_state.get("target_column", "target")

            if not have_data:
                st.info("Upload datasets in Step 1 (and set target) to compute feature importance.")
            else:
                # Build a signature of (dataset order, file bytes hash, target)
                ds_names = [n for n in st.session_state.selected_datasets if n in st.session_state.uploaded_files_map]
                sig_items, files_to_run = [], []
                for name in ds_names:
                    fobj = st.session_state.uploaded_files_map[name]
                    try: fobj.seek(0)
                    except Exception: pass
                    sig_items.append((name, _bytesig_of_upload(fobj)))
                    files_to_run.append(fobj)
                current_signature = (tuple(sig_items), target)

                # Detect input change; mark as stale but DO NOT recompute
                if st.session_state.fi_signature is not None and st.session_state.fi_signature != current_signature:
                    st.session_state.fi_stale = True

                # Button: only this triggers computation
                if st.button("Compute Feature Importance (per paper)", key="btn_fi_compute"):
                    try:
                        with st.spinner("Computing feature importance..."):
                            fi_results = compute_feature_importance_for_files(files_to_run, target=target)
                        st.session_state.fi_results_cache = fi_results
                        st.session_state.fi_signature = current_signature
                        st.session_state.fi_stale = False
                        st.success("Feature importance computed.")
                    except Exception as e:
                        st.error(f"Failed to compute feature importance: {e}")

                # Show stale notice if inputs changed since last compute
                if st.session_state.fi_stale:
                    st.warning("Inputs changed since last compute. Results below are from the previous run. Press the button to refresh.")

                # Display cached results (persist across any UI change)
                if st.session_state.fi_results_cache:
                    for ds_name, payload in st.session_state.fi_results_cache.items():
                        st.markdown(f"#### Dataset: `{ds_name}` — Top 20 (merged RF/LR)")
                        meta = payload.get("meta", {})
                        st.caption(
                            f"Rows: {meta.get('n_rows')}, Columns: {meta.get('n_cols')}, "
                            f"Kept after missing-drop: {len(meta.get('kept_columns_after_missing_drop', []))}"
                        )
                        st.dataframe(payload["merged"].head(20))

                        # 🚫 Do NOT use expanders inside an expander.
                        # ✅ Use tabs instead:
                        t1, t2 = st.tabs([
                            f"RandomForest importance (full) — {ds_name}",
                            f"LogisticRegression L1 |coef| (full) — {ds_name}",
                        ])
                        with t1:
                            st.dataframe(payload["rf"])
                        with t2:
                            st.dataframe(payload["lr"])
                else:
                    st.info("No feature-importance results yet. Click the button to compute.")


        # after the Step 1.25 expander block
        self._render_step_1_4_feature_selector()
        st.markdown("---")

        # --- Steps 1.5, 2, 3, 4, 5 (Conditional) ---
        if st.session_state.get("selected_datasets"):
            with st.expander("Step 1.5: Preprocessing Options", expanded=False):
                self._render_step_1_5_preprocessing_options()
                self._display_step_1_5_results()
            st.markdown("---")

            with st.expander("Step 2: Select Models & Target", expanded=False):
                self._render_step_2_model_selection()
                self._display_step_2_results()
            st.markdown("---")

            with st.expander("Step 3: Run Experiment", expanded=False):
                if st.session_state.get("target_column") and st.session_state.get("selected_models"):
                    self._render_step_3_run_experiment()
                    self._display_step_3_results()
                else:
                    st.info("Complete Step 2 (select target and models) to run the experiment.")

            with st.expander("Step 4: Benchmark Analysis", expanded=False):
                self._render_step_4_benchmark_analysis()
                self._display_step_4_results()

            st.markdown("---")
            # Step 5: No expander to prevent collapse on widget interaction
            if st.session_state.get("benchmark_results_df") is not None:
                self._render_step_5_shap_analysis()
            else:
                st.header("Step 5: Global SHAP & Reliability Analysis")
                st.info("Run Step 4 to identify benchmark models before running Global SHAP.")
            
            # --- Step 5.5: Reliability Test (separate) ---
            st.markdown("---")
            if st.session_state.get("benchmark_results_df") is not None or st.session_state.get("selected_datasets"):
                self._render_step_5_5_reliability_test()
            else:
                st.header("Step 5.5: Reliability Test for SHAP (rank stability & sanity)")
                st.info("Run Step 3/4 to produce benchmark models (or upload datasets) before running reliability tests.")

            # --- Step 6: Local SHAP Analysis ---
            st.markdown("---")
            if st.session_state.get("benchmark_results_df") is not None:
                self._render_step_6_local_analysis()
            else:
                # Still show the header, but with an info box
                st.header("Step 6: Local SHAP Analysis (Explain a Single Row)")
                st.info("Run Step 4 to identify benchmark models before running Local SHAP.")
                
        else:
            st.info("Complete Step 1 (upload datasets) to proceed.")

        # Fire the expander-opening script after all sections are rendered
        fire_expand_all_if_pending()
            


def main() -> None:
    """
    Main function to instantiate and run the Streamlit app.
    """
    app = ExperimentSetupApp()
    app.run()

if __name__ == "__main__":
    main()

