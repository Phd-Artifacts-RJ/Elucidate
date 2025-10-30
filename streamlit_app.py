# streamlit_app.py
from distro import name
import streamlit as st
# Prefer wide layout for the app by default
st.set_page_config(layout="wide")
import pandas as pd
import numpy as np  # Added for benchmark calculations
from typing import List, Dict, Any
from io import StringIO # Added for reading full DF
from dotenv import load_dotenv # --- NEW: Import dotenv ---
from feature_importance_paper import compute_feature_importance_for_files
import hashlib, io
from io import BytesIO
from pathlib import Path
import matplotlib.pyplot as plt

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
    from shap_analysis import get_shap_values, get_local_shap_explanation
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
            st.session_state.uploaded_files_map: Dict[str, Any] = {} # Stores the actual UploadedFile objects
        if 'selected_datasets' not in st.session_state:
            st.session_state.selected_datasets: List[str] = [] # Stores just the names
        
        if 'use_smote' not in st.session_state:
            st.session_state.use_smote: bool = False
            
        if 'target_column' not in st.session_state:
            st.session_state.target_column: str = "target"
        if 'selected_model_groups' not in st.session_state:
            st.session_state.selected_model_groups: List[str] = []
        if 'selected_models' not in st.session_state:
            st.session_state.selected_models: List[str] = []
        
        if 'results' not in st.session_state:
            st.session_state.results: Dict[str, Any] = {} 
        
        if 'benchmark_results_df' not in st.session_state:
            st.session_state.benchmark_results_df = None # Will store a DataFrame
        if 'benchmark_auc_comparison' not in st.session_state:
            st.session_state.benchmark_auc_comparison = None 

        if 'run_shap' not in st.session_state:
            st.session_state.run_shap: bool = False
            
        # --- NEW: Cache for full DataFrames for Step 6 ---
        if 'full_dfs' not in st.session_state:
            st.session_state.full_dfs: Dict[str, pd.DataFrame] = {}

        if 'feature_selection' not in st.session_state:
            # { dataset_name: [list of selected feature columns] }
            st.session_state.feature_selection: Dict[str, List[str]] = {}

        if "fi_results_cache" not in st.session_state:
            # { dataset_name: payload }, same structure you already display (rf/lr/merged/meta)
            st.session_state.fi_results_cache = {}

        if "fi_signature" not in st.session_state:
            # tuple that identifies what the cache corresponds to (files+target)
            st.session_state.fi_signature = None

        # run once early in the app
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

                    except Exception as e:
                        st.warning(f"Could not produce preview for {name}: {e}")

                    # (retain your existing target value-counts block below, unchanged)
                    target = st.session_state.get('target_column', 'target')
                    try:
                        try:
                            fileobj.seek(0)
                        except Exception:
                            pass
                        header = pd.read_csv(fileobj, nrows=0)
                        cols = header.columns.tolist()
                        if target in cols:
                            counts = {}
                            try:
                                fileobj.seek(0)
                            except Exception:
                                pass
                            for chunk in pd.read_csv(fileobj, usecols=[target], chunksize=100_000):
                                vc = chunk[target].value_counts(dropna=False)
                                for k, v in vc.items():
                                    counts[k] = counts.get(k, 0) + int(v)
                            if counts:
                                counts_series = pd.Series(counts).sort_values(ascending=False)
                                st.markdown(f"**Value counts (`{target}`)**")
                                st.write(counts_series.to_frame(name="count"))
                        else:
                            st.info(f"Target column '{target}' not found in this file.")
                    except Exception as e:
                        st.warning(f"Could not compute value counts for {name}: {e}")


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
                        ["—", "Top 10", "Top 15", "Top 20"],
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
                    top_lookup = {"Top 10": 10, "Top 15": 15, "Top 20": 20}
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
        if st.session_state.use_smote:
            st.info("SMOTE (Oversampling) is **Enabled**.")
        else:
            st.info("SMOTE (Oversampling) is **Disabled**.")


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
        Displays the avg. AUC comparison tables and the final benchmark results.
        """
        auc_comp = st.session_state.get("benchmark_auc_comparison") or {}
        bench_df = st.session_state.get("benchmark_results_df")

        if auc_comp:
            st.subheader("Average AUC Comparison (by Group)")
            st.markdown("This table shows the average AUC for all models across all datasets, which is used to select the benchmark model for each group.")
            for group_name, auc_df in auc_comp.items():
                st.markdown(f"#### Group: {group_name}")
                st.dataframe(auc_df.style.format("{:.4f}"))
            st.markdown("---")

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
        elif not auc_comp:
            st.info("Run benchmark analysis to see the final summary table here.")


    def _render_step_5_shap_analysis(self):
        """
        Renders the button and logic for SHAP analysis.
        """
        st.header("Step 5: Global SHAP Analysis") # Renamed title
        
        if not SHAP_AVAILABLE:
            st.error("SHAP library not found. Please install it to run this analysis: `pip install shap matplotlib`")
            return

        st.markdown("Generate SHAP summary plots (global feature importance) for the best-performing **benchmark model** from each dataset.")
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

            with st.spinner("Generating Global SHAP plots... This may take several minutes."):
                for index, row in benchmark_df.iterrows():
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

                        if model_to_explain is None or X_train is None or X_test is None:
                            st.warning(f"Could not find stored model or data for {dataset}. Skipping.")
                            continue
                        
                        # --- NEW PLOTTING LOGIC ---
                        
                        # 1. Get the data (this is the slow part)
                        # Unified helper now returns: (sv: shap.Explanation, explain_df: DataFrame, shap_global_df: DataFrame)
                        sv, explain_data_sample_df, shap_global_df = get_shap_values(
                            model_to_explain, X_train, X_test
                        )
                        
                        # --- ADD THIS BLOCK ---
                        # Cache the global SHAP df for Step 6
                        if 'global_shap_dfs' not in st.session_state:
                            st.session_state.global_shap_dfs = {}
                        st.session_state.global_shap_dfs[dataset] = shap_global_df
                        # --- END BLOCK ---

                        # 2. Create two columns for the plots
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("##### Summary Plot (Bar)")
                            st.caption("Average impact (magnitude) of each feature.")
                            fig, ax = plt.subplots()
                            # Bar
                            shap.summary_plot(sv, explain_data_sample_df, plot_type="bar", show=False)
                            st.pyplot(fig)
                            plt.close(fig)

                        with col2:
                            st.markdown("##### Summary Plot (Dot)")
                            st.caption("Distribution of feature impacts (magnitude and direction).")
                            fig, ax = plt.subplots()
                            # Dot
                            shap.summary_plot(sv, explain_data_sample_df, show=False)
                            st.pyplot(fig)
                            plt.close(fig)
                    
                    except Exception as e:
                        st.error(f"Failed to generate SHAP plot for {dataset} - {model_name}: {e}")


    def _render_step_5_2_reliability(self):
        st.header("Step 5.2: Reliability Checks")
        bench_df = st.session_state.get("benchmark_results_df")
        all_results = st.session_state.get("results", {})
        if bench_df is None or not all_results:
            st.info("Run Step 4 first.")
            return

        ds = st.selectbox("Dataset", st.session_state.selected_datasets, key="rel_ds")
        if not ds: 
            return
        row = bench_df[bench_df['Dataset'] == ds]
        if row.empty:
            st.warning("No benchmark for this dataset.")
            return
        group = row.iloc[0]['Model Group']; model_name = row.iloc[0]['Benchmark Model']
        model = all_results.get(ds, {}).get('models', {}).get(group, {}).get(model_name)
        data  = all_results.get(ds, {}).get('data', {})
        X_train = data.get('X_train'); X_test = data.get('X_test')
        y_train = data.get('y_train')  # ✅

        # (Optional) if you want true labels, keep y in results in models.py

        c1, c2, c3 = st.columns(3)
        n_trials = c1.number_input("Stability trials", 5, 50, 10)
        n_bg = c2.number_input("Background size", 50, 2000, 200, step=50)
        run = c3.button("Run reliability")

        if run:
            with st.spinner("Computing SHAP rank stability..."):
                stab = shap_rank_stability(model, X_train, X_test, n_bg_samples=n_bg, n_trials=int(n_trials))
            st.subheader("Rank stability (lower std_rank is better)")
            st.dataframe(stab)

            with st.spinner("Running model randomization sanity test..."):
                # Guard: ensure y_train is present and valid before calling the SHAP sanity function
                if y_train is None:
                    st.info("Skipping SHAP randomization sanity check: y_train not available in stored results.")
                    ratio = None
                else:
                    try:
                        # Accept array-like y_train (the function will still validate/convert)
                        ratio = model_randomization_sanity(model, X_train, X_test, y_train, n_bg_samples=n_bg)
                    except Exception as e:
                        st.warning(f"Sanity check failed: {e}")
                        ratio = None

            if ratio is not None:
                st.info(f"Sanity ratio (orig mass / randomized mass): **{ratio:.2f}**. Expect this ≫ 1.0 if SHAP is meaningful.")
            else:
                st.info("Sanity ratio unavailable.")

    # --- NEW: Step 6 Render Logic ---
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
                shap.waterfall_plot(explanation, max_display=12, show=False)
                st.pyplot(fig)
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
                        st.pyplot(fig); plt.close(fig)
            # --- END: NEW INTERACTIONS & PDP BLOCK ---

        except Exception as e:
            st.error(f"Failed to generate local SHAP plot: {e}")
            st.exception(e) # Show full traceback
    def run(self):
        """
        Run the main application logic and render the UI.
        """
        
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
            with st.expander("Step 5: Global SHAP Analysis", expanded=False):
                if st.session_state.get("benchmark_results_df") is not None:
                    self._render_step_5_shap_analysis()
                else:
                    st.info("Run Step 4 to identify benchmark models before running Global SHAP.")

            with st.expander("Step 5.2: Reliability Checks", expanded=False):
                if st.session_state.get("benchmark_results_df") is not None:
                    self._render_step_5_2_reliability()
                else:
                    st.info("Run Step 4 to identify benchmark models before reliability checks.")

            # --- FIX: Step 6 is no longer in an expander ---
            st.markdown("---")
            if st.session_state.get("benchmark_results_df") is not None:
                self._render_step_6_local_analysis()
            else:
                # Still show the header, but with an info box
                st.header("Step 6: Local SHAP Analysis (Explain a Single Row)")
                st.info("Run Step 4 to identify benchmark models before running Local SHAP.")
                
        else:
            st.info("Complete Step 1 (upload datasets) to proceed.")


def main() -> None:
    """
    Main function to instantiate and run the Streamlit app.
    """
    app = ExperimentSetupApp()
    app.run()

if __name__ == "__main__":
    main()

