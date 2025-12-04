# models.py
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    FunctionTransformer
)
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from io import StringIO

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, make_scorer


# --- NEW IMPORTS FOR KNN, XGBOOST, LIGHTGBM ---
from sklearn.neighbors import KNeighborsClassifier

# deep learning models (PyTorch-backed)
try:
    from deep_models import TorchMLPClassifier
    TORCH_MODELS_AVAILABLE = True
except ImportError:
    TORCH_MODELS_AVAILABLE = False
    TorchMLPClassifier = None
    print("Warning: deep_models.py or torch not available. DL models disabled.")

# XGBoost and LightGBM imports with fallbacks
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    class XGBClassifier: # Placeholder
        def __init__(self, **kwargs): pass
    print("Warning: xgboost not found. XGBoost models will be disabled.")

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    class LGBMClassifier: # Placeholder
        def __init__(self, **kwargs): pass
    print("Warning: lightgbm not found. LightGBM models will be disabled.")

# --- NEW IMPORTS FOR SMOTE ---
# We use a try/except block in case imbalanced-learn is not installed
try:
    from imblearn.over_sampling import SMOTE
    # We must use the imblearn pipeline to correctly apply SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    # Create a placeholder class if imblearn is not available
    # This prevents the app from crashing if SMOTE is selected
    class SMOTE:
        def __init__(self, **kwargs):
            pass
    class ImbPipeline:
        def __init__(self, steps):
            pass
    print("Warning: imbalanced-learn not found. SMOTE functionality will be disabled.")

# at the top of models.py
try:
    from deep_models import (
        TorchMLPClassifier,
        TorchTCNClassifier,
        TorchTransformerEncoderClassifier,
    )
    TORCH_MODELS_AVAILABLE = True
except ImportError:
    TORCH_MODELS_AVAILABLE = False
    TorchMLPClassifier = TorchTCNClassifier = TorchTransformerEncoderClassifier = None
    print("Warning: deep_models.py or torch not available. DL models disabled.")


# Import the metrics functions from your metrics.py file
try:
    # --- MODIFIED: Added F1 and Recall to placeholder ---
    from metrics import compute_metrics
except ImportError:
    print("Warning: metrics.py not found. Using placeholder metrics.")
    # Define a placeholder if metrics.py is missing, to avoid crashing
    def compute_metrics(y_tr, y_te, p_te, **kwargs):
        return {"AUC": 0.5, "PCC": 0.5, "F1": 0.5, "Recall": 0.5, "BS": 0.25, "KS": 0.0, "PG": 0.0, "H": 0.0}

def _preprocessor():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        
        # --- FIX: Convert all categorical columns to string type ---
        # This solves the "Got ['float', 'str']" error by ensuring
        # the OneHotEncoder only ever receives string data.
        ("string_caster", FunctionTransformer(lambda x: x.astype(str))),
        
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric, make_column_selector(dtype_include=np.number)),
            ("cat", categorical, make_column_selector(dtype_exclude=np.number)),
        ]
    )

# ---------- Individual Classifiers ----------

# ---------- Logistic Regression ----------

def build_lr_lbfgs():
    base = LogisticRegression(
        penalty="l2",
        C=1e6,
        solver="lbfgs", # Corrected from 'lfgs'
        max_iter=1000,
        random_state=42,
        n_jobs=None,     # supports parallelization via joblib env vars
    )
    return make_pipeline(_preprocessor(), CalibratedClassifierCV(base, method="sigmoid", cv=3))

def build_lr_saga():
    base = LogisticRegression(
        penalty="l2",
        C=1e6,
        solver="saga",
        max_iter=2000,   # saga often needs more iterations
        random_state=42,
    )
    return make_pipeline(_preprocessor(), CalibratedClassifierCV(base, method="sigmoid", cv=3))

def build_lr_newton_cg():
    base = LogisticRegression(
        penalty="l2",
        C=1e6,
        solver="newton-cg",
        max_iter=1000,
        random_state=42,
    )
    return make_pipeline(_preprocessor(), CalibratedClassifierCV(base, method="sigmoid", cv=3))

# ---------- Regularized Logistic Regression ----------

def build_lr_reg_saga():
    """Regularised LogisticRegression using saga (supports elasticnet)."""
    base = LogisticRegression(
        penalty="elasticnet",
        l1_ratio=0.5,
        C=1.0,
        solver="saga",
        max_iter=2000,
        random_state=42,
    )
    return make_pipeline(_preprocessor(), CalibratedClassifierCV(base, method="sigmoid", cv=3))

def build_lr_reg_lbfgs():
    """Regularised LogisticRegression using lbfgs (L2 penalty)."""
    base = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
        n_jobs=None,
    )
    return make_pipeline(_preprocessor(), CalibratedClassifierCV(base, method="sigmoid", cv=3))

def build_lr_reg_liblinear():
    """Regularised LogisticRegression using liblinear (supports L1/L2, ovR)."""
    base = LogisticRegression(
        penalty="l1",
        C=1.0,
        solver="liblinear",
        max_iter=1000,
        random_state=42,
    )
    return make_pipeline(_preprocessor(), CalibratedClassifierCV(base, method="sigmoid", cv=3))


# --- NEW: AdaBoost (Alternating Decision Tree) Group ---

def build_adaboost(n_estimators):
    """
    Helper function to build an AdaBoost classifier with a Decision Tree stump.
    This is scikit-learn's equivalent of an alternating decision tree.
    """
    base = AdaBoostClassifier(
        # Use a "stump" (max_depth=1) as the weak learner
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=n_estimators,
        random_state=42,
        algorithm='SAMME' # Default, supports predict_proba
    )
    # We calibrate it for consistency with other models
    return make_pipeline(_preprocessor(), CalibratedClassifierCV(base, method="sigmoid", cv=3))

def build_adaboost_10():
    return build_adaboost(n_estimators=10)

def build_adaboost_20():
    return build_adaboost(n_estimators=20)

def build_adaboost_30():
    return build_adaboost(n_estimators=30)

# =========================
# HOMOGENEOUS ENSEMBLES (with Lessmann-style grids)
# =========================
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

# -------- Bagged Decision Trees (Bag) --------
def make_bag_cart(n_estimators: int):
    base_cart = DecisionTreeClassifier(criterion="gini", random_state=42)
    ensemble = BaggingClassifier(
        estimator=base_cart,
        n_estimators=int(n_estimators),
        random_state=42
    )
    return make_pipeline(_preprocessor(), CalibratedClassifierCV(ensemble, method="sigmoid", cv=3))

BAG_CART_CANDIDATES = [10, 20, 50, 100, 250, 500, 1000]
BAG_CART_BUILDERS = {f"bag_cart_{n}": (lambda n=n: make_bag_cart(n)) for n in BAG_CART_CANDIDATES}

# -------- Bagged MLP (BagNN) --------
def make_bagnn(n_estimators: int):
    base_mlp = MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=1000,
        early_stopping=True,
        random_state=42,
    )
    ensemble = BaggingClassifier(
        estimator=base_mlp,
        n_estimators=int(n_estimators),
        random_state=42
    )
    return make_pipeline(_preprocessor(), CalibratedClassifierCV(ensemble, method="sigmoid", cv=3))

BAGNN_CANDIDATES = [5, 10, 25, 100]
BAGNN_BUILDERS = {f"bagnn_{n}": (lambda n=n: make_bagnn(n)) for n in BAGNN_CANDIDATES}

# -------- Boosted Decision Trees (Boost / AdaBoost variants) --------
# Paper varies iterations and learning rate; we use SAMME (smooth proba).
def make_adaboost_iters_lr(n_estimators: int, learning_rate: float):
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=42),  # stumps
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        algorithm="SAMME",
        random_state=42,
    )
    return make_pipeline(_preprocessor(), CalibratedClassifierCV(ada, method="sigmoid", cv=3))

ADABOOST_ITERS = [10, 50, 100, 250, 500, 1000]
ADABOOST_LRS = [0.1, 0.5, 1.0]
ADABOOST_BUILDERS = {
    f"boost_dt_{n}x{lr}".replace('.', 'p'): (lambda n=n, lr=lr: make_adaboost_iters_lr(n, lr))
    for n in ADABOOST_ITERS for lr in ADABOOST_LRS
}

# -------- Random Forest (RF) --------
# Paper: n_estimators in {100,250,500,750,1000}; "mtry" around sqrt(m)*k.
# sklearn uses 'max_features' as fraction or 'sqrt'. We approximate the grid.
def make_rf(n_estimators: int, max_features):
    rf = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_features=max_features,   # 'sqrt' or fraction in (0,1]
        bootstrap=True,
        random_state=42,
    )
    return make_pipeline(_preprocessor(), CalibratedClassifierCV(rf, method="sigmoid", cv=3))

RF_TREES = [100, 250, 500, 750, 1000]
RF_MAXF = ['sqrt', 0.1, 0.25, 0.5, 1.0]  # approximation to paper’s √m scaling
RF_BUILDERS = {
    f"rf_{n}_mf_{str(mf).replace('.', 'p')}": (lambda n=n, mf=mf: make_rf(n, mf))
    for n in RF_TREES for mf in RF_MAXF
}

# -------- Stochastic Gradient Boosting (SGB) --------
def make_sgb(n_estimators: int):
    sgb = GradientBoostingClassifier(
        n_estimators=int(n_estimators),
        subsample=0.7,      # stochastic component
        random_state=42
    )
    return make_pipeline(_preprocessor(), CalibratedClassifierCV(sgb, method="sigmoid", cv=3))

SGB_ITERS = [10, 20, 50, 100, 250, 500, 1000]
SGB_BUILDERS = {f"sgb_{n}": (lambda n=n: make_sgb(n)) for n in SGB_ITERS}


# -------- K-Nearest Neighbors (K-NN) --------

def _make_calibrated_knn(best_params):
    knn = KNeighborsClassifier(
        n_neighbors=best_params['knn__n_neighbors'],
        weights=best_params['knn__weights'],
        metric=best_params['knn__metric'],
        p=best_params['knn__p'],
    )
    try:
        # scikit-learn >= 1.4
        return CalibratedClassifierCV(estimator=knn, method="sigmoid", cv=3)
    except TypeError:
        # scikit-learn <= 1.3
        return CalibratedClassifierCV(base_estimator=knn, method="sigmoid", cv=3)


def make_knn(n_neighbors: int):
    knn = KNeighborsClassifier(
        n_neighbors=int(n_neighbors)
    )
    # We calibrate KNN as its probability estimates can be noisy
    return make_pipeline(_preprocessor(), CalibratedClassifierCV(knn, method="sigmoid", cv=3))

KNN_CANDIDATES = [3, 5, 7, 11]
KNN_BUILDERS = {f"knn_{n}": (lambda n=n: make_knn(n)) for n in KNN_CANDIDATES}

def build_knn_tuned():
    """Marker function for tuned KNN; actual tuning happens in run_experiment."""
    return None

KNN_BUILDERS["knn_tuned"] = build_knn_tuned
# -------- Feature Importance Helpers --------

def _tune_knn_estimator(X_train, y_train, use_smote: bool):
    """
    Tune KNN with stratified CV, AUC scoring, and SMOTE applied *within* each CV fold
    (no leakage). Returns a final *pipeline* that includes the chosen KNN hyperparams
    and calibration, plus a dict of {best_params, cv_auc}.
    """
    if use_smote and not IMBLEARN_AVAILABLE:
        raise RuntimeError("SMOTE requested but 'imbalanced-learn' is not installed.")

    # Build the CV pipeline: preprocess -> (optional SMOTE) -> KNN
    steps = [('prep', _preprocessor())]
    if use_smote:
        steps.append(('smote', SMOTE(random_state=42)))
    steps.append(('knn', KNeighborsClassifier()))
    cv_pipe = ImbPipeline(steps)  # ImbPipeline works also when SMOTE is absent

    # Rival hypotheses we want to test: local vs global (k), weighting, geometry
    param_grid = {
        'knn__n_neighbors': [3, 5, 7, 11, 15, 21],
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['minkowski'],
        'knn__p': [1, 2],      # 1=Manhattan, 2=Euclidean
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(roc_auc_score, needs_threshold=True)

    grid = GridSearchCV(
        estimator=cv_pipe,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        refit=True,  # keep best CV pipeline
    )
    grid.fit(X_train, y_train)

    best_params = grid.best_params_
    cv_auc = grid.best_score_

    # Build the *final* pipeline for training/evaluation:
    # preprocess -> (optional SMOTE) -> Calibrated(KNN(best_params))
    # We keep SMOTE *inside* the pipeline to avoid leakage on refit.
    final_steps = [('prep', _preprocessor())]
    if use_smote:
        final_steps.append(('smote', SMOTE(random_state=42)))
    final_steps.append(('cal', _make_calibrated_knn(best_params)))
    final_pipe = ImbPipeline(final_steps)

    return final_pipe, {'BestParams': best_params, 'CV_AUC': cv_auc}

# -------- XGBoost (XGB) --------
def make_xgb(n_estimators: int, learning_rate: float):
    xgb = XGBClassifier(
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        random_state=42,
        use_label_encoder=False, # Suppress warning
        eval_metric='logloss'    # Suppress warning
    )
    return make_pipeline(_preprocessor(), CalibratedClassifierCV(xgb, method="sigmoid", cv=3))

XGB_ESTIMATORS = [100, 200, 300]
XGB_LRS = [0.1, 0.3]
XGB_BUILDERS = {}
if XGB_AVAILABLE:
    XGB_BUILDERS = {
        f"xgb_{n}x{lr}".replace('.', 'p'): (lambda n=n, lr=lr: make_xgb(n, lr))
        for n in XGB_ESTIMATORS for lr in XGB_LRS
    }

# -------- LightGBM (LGBM) --------
def make_lgbm(n_estimators: int, learning_rate: float):
    lgbm = LGBMClassifier(
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        random_state=42
    )
    return make_pipeline(_preprocessor(), CalibratedClassifierCV(lgbm, method="sigmoid", cv=3))

LGBM_ESTIMATORS = [100, 200, 300]
LGBM_LRS = [0.1, 0.3]
LGBM_BUILDERS = {}
if LGBM_AVAILABLE:
    LGBM_BUILDERS = {
        f"lgbm_{n}x{lr}".replace('.', 'p'): (lambda n=n, lr=lr: make_lgbm(n, lr))
        for n in LGBM_ESTIMATORS for lr in LGBM_LRS
    }

# -------- PyTorch MLP (via TorchMLPClassifier) --------

def build_torch_mlp():
    if not TORCH_MODELS_AVAILABLE:
        raise RuntimeError("Torch models not available.")
    est = TorchMLPClassifier(
        hidden_dims=(64, 32),
        dropout=0.1,
        lr=1e-3,
        batch_size=256,
        max_epochs=50,
        patience=5,
        verbose=True,
        random_state=42,
    )
    return make_pipeline(_preprocessor(), est)

def build_torch_tcn():
    if not TORCH_MODELS_AVAILABLE:
        raise RuntimeError("Torch models not available.")
    est = TorchTCNClassifier(
        seq_len=1,          # match your rolling window / years
        channels=(64, 64, 64),
        kernel_size=3,
        dropout=0.1,
        lr=1e-3,
        batch_size=128,
        max_epochs=50,
        patience=5,
        verbose=True,
        random_state=42,
    )
    return make_pipeline(_preprocessor(), est)

def build_torch_transformer():
    if not TORCH_MODELS_AVAILABLE:
        raise RuntimeError("Torch models not available.")
    est = TorchTransformerEncoderClassifier(
        seq_len=1,          # set this to whatever matches your sequence construction
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        lr=1e-3,
        batch_size=128,
        max_epochs=50,
        patience=5,
        verbose=True,
        random_state=42,
    )
    return make_pipeline(_preprocessor(), est)

# Dictionary of model groups for the Streamlit app
MODELS = {
    "lr": {
        "lr_lbfgs": build_lr_lbfgs,
        "lr_saga": build_lr_saga,
        "lr_newton_cg": build_lr_newton_cg,
    },
    "lr_reg": {
        "lr_reg_saga": build_lr_reg_saga,
        "lr_reg_lbfgs": build_lr_reg_lbfgs,
        "lr_reg_liblinear": build_lr_reg_liblinear,
    },
    # --- NEWLY ADDED GROUP ---
    "adaboost": {
        "adaboost_10": build_adaboost_10,
        "adaboost_20": build_adaboost_20,
        "adaboost_30": build_adaboost_30,
    },
}

# =========================
# MODELS REGISTRY (extended)
# =========================
MODELS.update({
    "Bag-CART": BAG_CART_BUILDERS,     # Bagged decision trees
    "BagNN": BAGNN_BUILDERS,           # Bagged MLP
    "Boost-DT": ADABOOST_BUILDERS,     # AdaBoost (iterations × learning rate)
    "RF": RF_BUILDERS,                 # Random forest grid (trees × max_features)
    "SGB": SGB_BUILDERS,               # Stochastic gradient boosting
    
    # --- ADD THESE NEW LINES ---
    "KNN": KNN_BUILDERS,               # K-Nearest Neighbors
    "XGB": XGB_BUILDERS,               # XGBoost
    "LGBM": LGBM_BUILDERS,             # LightGBM
})

# if TORCH_MODELS_AVAILABLE:
#     MODELS["DL"] = {
#         "torch_mlp": build_torch_mlp,
#     }

if TORCH_MODELS_AVAILABLE:
    MODELS["DL"] = {
        "torch_mlp": build_torch_mlp,
        "torch_tcn": build_torch_tcn,
        "torch_transformer": build_torch_transformer,
    }

# --- MODIFIED EXECUTION FUNCTION ---

def run_experiment(uploaded_files, target_column, selected_model_groups_dict, use_smote):
    """
    Runs the full experiment on all uploaded datasets and selected models.

    Args:
        uploaded_files (list): List of Streamlit UploadedFile objects.
        target_column (str): The name of the target variable.
        selected_model_groups_dict (dict): A dictionary of the selected model groups.
        use_smote (bool): Whether to apply SMOTE to the training data.
        
    Returns:
        A dictionary structured as:
        {
            dataset_name: {
                "metrics": {group: {model: {metric: value, ...}}},
                "models": {group: {model: fitted_pipeline_object}},
                "data": {"X_train": X_train_df, "X_test": X_test_df},
                "error": "..."
            },
            ...
        }
    """
    results = {}

    if use_smote and not IMBLEARN_AVAILABLE:
        return {"error": "SMOTE was selected, but the 'imbalanced-learn' library is not installed."}

    for file in uploaded_files:
        dataset_name = file.name
        # --- NEW: Modified results structure ---
        results[dataset_name] = {
            "metrics": {},
            "models": {},
            "data": {},
            "error": None
        }
        
        try:
            # Read the uploaded file into a pandas DataFrame
            # We use StringIO to read the file buffer
            stringio = StringIO(file.getvalue().decode("utf-8"))
            df = pd.read_csv(stringio)

            # Check if target column exists
            if target_column not in df.columns:
                results[dataset_name]["error"] = f"Target column '{target_column}' not found."
                continue

            # --- Data Preparation ---
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Simple train-test split (you could replace this with cross-validation)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # --- NEW: Store data for SHAP & reliability ---
            results[dataset_name]["data"] = {
                "X_train": X_train, "X_test": X_test,
                "y_train": y_train, "y_test": y_test
            }

            # --- Model Training and Evaluation ---
            for group_name, models in selected_model_groups_dict.items():
                # --- NEW: Initialize nested dicts ---
                results[dataset_name]["metrics"][group_name] = {}
                results[dataset_name]["models"][group_name] = {}
                
                for model_name, model_builder in models.items():
                    try:

                        # --- BEGIN: tuned KNN branch ---
                        if group_name == "KNN" and model_name == "knn_tuned":
                            # Run proper CV tuning with SMOTE-in-fold if requested
                            tuned_pipe, tuning_info = _tune_knn_estimator(X_train, y_train, use_smote)
                            # Fit on full training set (SMOTE & calibration are in-pipeline)
                            tuned_pipe.fit(X_train, y_train)

                            # Evaluate on the untouched test split
                            y_pred_proba = tuned_pipe.predict_proba(X_test)[:, 1]
                            model_metrics = compute_metrics(y_train, y_test, y_pred_proba)

                            # Attach tuning diagnostics
                            model_metrics.update(tuning_info)

                            results[dataset_name]["metrics"][group_name][model_name] = model_metrics
                            results[dataset_name]["models"][group_name][model_name] = tuned_pipe
                            continue
                        # --- END: tuned KNN branch ---

                        # Build the original pipeline (preprocessor + classifier)
                        original_pipeline = model_builder()
                        
                        # --- NEW SMOTE LOGIC ---
                        if use_smote:
                            # If SMOTE is selected, we rebuild the pipeline
                            # using imblearn's Pipeline to inject SMOTE
                            
                            # Extract the steps from the original pipeline
                            original_steps = original_pipeline.steps
                            preprocessor = original_steps[0] # ('columntransformer', ...)
                            classifier = original_steps[1]   # ('calibratedclassifiercv', ...)

                            # Create a new pipeline with SMOTE
                            pipeline = ImbPipeline([
                                preprocessor,
                                ('smote', SMOTE(random_state=42)),
                                classifier
                            ])
                        else:
                            # If SMOTE is not selected, use the original pipeline
                            pipeline = original_pipeline
                        # --- END NEW SMOTE LOGIC ---

                        # Train the model (this will now use SMOTE if enabled)
                        pipeline.fit(X_train, y_train)
                        
                        # Get predicted probabilities for the positive class
                        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
                        
                        # Compute all metrics
                        model_metrics = compute_metrics(y_train, y_test, y_pred_proba)
                        
                        # --- NEW: Store metrics and models separately ---
                        results[dataset_name]["metrics"][group_name][model_name] = model_metrics
                        results[dataset_name]["models"][group_name][model_name] = pipeline

                    except Exception as e:
                        # Store any error that occurs during model training/prediction
                        results[dataset_name]["metrics"][group_name][model_name] = {"error": str(e)}

        except Exception as e:
            # Store any error that occurs during file reading/processing
            results[dataset_name]["error"] = f"Failed to process file: {str(e)}"

    return results