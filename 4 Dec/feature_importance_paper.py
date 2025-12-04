# feature_importance_paper.py
"""
Paper-style feature-importance computation to insert before Step 1.5.

Implements the preprocessing and selectors described in the paper:
- Drop columns with >90% missing
- Impute numerics (mean) and categoricals (most_frequent)
- One-hot encode categoricals
- Min–max scale numerics
- Supervised selectors:
    * RandomForestClassifier: impurity-based feature importance
    * LogisticRegression (L1): absolute coefficient magnitude on scaled inputs

Notes:
- This is NOT an explanation tool. It ranks features by contribution within these
  models. Treat as conjectural and subject to criticism.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple
import io

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, FunctionTransformer
from imblearn.over_sampling import RandomOverSampler

def _build_preprocessor(
    df: pd.DataFrame,
    target: str,
    max_missing_ratio: float = 0.90,
) -> Tuple[ColumnTransformer, List[str]]:
    """
    Build a ColumnTransformer that:
      - drops columns with > max_missing_ratio missing
      - imputes numerics (mean) & categoricals (most_frequent)
      - one-hot encodes categoricals
      - min–max scales numerics
    Returns (preprocessor, kept_feature_names)
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")

    X = df.drop(columns=[target]).copy()

    # Drop columns with >90% missing
    missing_ratio = X.isna().mean()
    keep_cols = missing_ratio[missing_ratio <= max_missing_ratio].index.tolist()
    X = X[keep_cols]

    # Selectors
    num_selector = make_column_selector(dtype_include=np.number)
    cat_selector = make_column_selector(dtype_include=["object", "category", "bool"])

    # Pipelines
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", MinMaxScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # ensure uniform dtype per column, required by encoders
            ("to_str", FunctionTransformer(lambda X: X.astype(str))),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_selector),
            ("cat", cat_pipe, cat_selector),
        ],
        remainder="drop",
        sparse_threshold=0.0,  # always dense
    )

    return preprocessor, keep_cols


def _expanded_feature_names(
    preprocessor: ColumnTransformer, X_fit: pd.DataFrame
) -> List[str]:
    """
    After fitting the preprocessor, recover the transformed feature names.
    """
    feature_names: List[str] = []

    # Numeric names (pass-through via num pipeline)
    num_indices = preprocessor.transformers_[0][2]  # selector function
    num_cols = list(num_indices(X_fit)) if callable(num_indices) else list(num_indices)
    feature_names.extend(num_cols)  # MinMaxScaler keeps same names

    # Categorical names (OneHot)
    cat_indices = preprocessor.transformers_[1][2]
    cat_cols = list(cat_indices(X_fit)) if callable(cat_indices) else list(cat_indices)

    onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]

    # Ensure the OneHotEncoder has been fitted before accessing categories_
    if not hasattr(onehot, "categories_"):
        # Fit it temporarily on the categorical subset to populate categories_
        X_cat = X_fit[cat_cols].astype(str)
        onehot.fit(X_cat)

    # Build names like col=value
    feature_names.extend(
        f"{col}={cat}"
        for col, cats in zip(cat_cols, onehot.categories_)
        for cat in cats
    )

    return feature_names


def _rf_importance(
    X_tr: np.ndarray, y_tr: np.ndarray, feat_names: List[str], random_state: int = 42
) -> pd.DataFrame:
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, random_state=random_state, n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    imp = rf.feature_importances_
    df = pd.DataFrame({"feature": feat_names, "importance": imp})
    df["rank"] = df["importance"].rank(ascending=False, method="first").astype(int)
    df = df.sort_values(["importance", "feature"], ascending=[False, True]).reset_index(drop=True)
    df.insert(0, "selector", "RandomForest")
    return df


def _lr_importance(
    X_tr: np.ndarray, y_tr: np.ndarray, feat_names: List[str], random_state: int = 42
) -> pd.DataFrame:
    # L1 to induce sparsity; liblinear handles OVR for binary cleanly
    lr = LogisticRegression(
        penalty="l1", solver="liblinear", C=1.0, random_state=random_state, max_iter=2000
    )
    lr.fit(X_tr, y_tr)
    coef = np.ravel(lr.coef_)  # shape (1, n_features) for binary
    imp = np.abs(coef)
    df = pd.DataFrame({"feature": feat_names, "importance": imp})
    df["rank"] = df["importance"].rank(ascending=False, method="first").astype(int)
    df = df.sort_values(["importance", "feature"], ascending=[False, True]).reset_index(drop=True)
    df.insert(0, "selector", "LogisticRegression_L1")
    return df


def compute_feature_importance_from_df(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.25,
    random_state: int = 42,
    max_missing_ratio: float = 0.90,
) -> Dict[str, pd.DataFrame]:
    """
    End-to-end: preprocess -> train/test split -> train selectors -> return importance frames.
    Returns dict with keys: 'rf', 'lr', 'merged'
    """
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found.")

    y = df[target].values
    preprocessor, kept = _build_preprocessor(df, target, max_missing_ratio)

    # Fit preprocessor on full X to materialize feature names deterministically
    X = df.drop(columns=[target])[kept]
    preprocessor.fit(X)
    all_feat_names = _expanded_feature_names(preprocessor, X)

    # Now do a proper split and transform
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_tr = preprocessor.transform(X_train)

    # Optional: perform random oversampling on the training set
    ros = RandomOverSampler(random_state=random_state)
    X_tr, y_train = ros.fit_resample(X_tr, y_train)

    # Train selectors
    rf_df = _rf_importance(X_tr, y_train, all_feat_names, random_state=random_state)
    lr_df = _lr_importance(X_tr, y_train, all_feat_names, random_state=random_state)

    # Merge (outer) with average normalized score for a single consolidated view
    merged = rf_df[["feature", "importance"]].rename(columns={"importance": "rf_importance"}).merge(
        lr_df[["feature", "importance"]].rename(columns={"importance": "lr_coef_abs"}),
        on="feature",
        how="outer",
    )
    # Fill NaNs with 0 (feature dropped by L1 or not used)
    merged[["rf_importance", "lr_coef_abs"]] = merged[["rf_importance", "lr_coef_abs"]].fillna(0.0)

    # Normalize per column to [0,1] for comparability, then compute mean score
    def _minmax(s: pd.Series) -> pd.Series:
        lo, hi = float(s.min()), float(s.max())
        if hi <= lo:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - lo) / (hi - lo)

    merged["rf_norm"] = _minmax(merged["rf_importance"])
    merged["lr_norm"] = _minmax(merged["lr_coef_abs"])
    merged["avg_score"] = merged[["rf_norm", "lr_norm"]].mean(axis=1)
    merged = merged.sort_values(["avg_score", "rf_norm", "lr_norm"], ascending=False).reset_index(drop=True)
    merged.insert(0, "rank", np.arange(1, len(merged) + 1))

    return {
        "rf": rf_df,
        "lr": lr_df,
        "merged": merged,
        "preprocessor": preprocessor,
        "feature_names": all_feat_names,
        "kept_columns_after_missing_drop": kept,
    }


def compute_feature_importance_for_files(
    uploads: List[Any],  # streamlit UploadedFile or file-like
    target: str,
    random_state: int = 42,
    max_missing_ratio: float = 0.90,
) -> Dict[str, Dict[str, Any]]:
    """
    Convenience wrapper for Streamlit integration.
    Returns: { dataset_name: { 'rf': df, 'lr': df, 'merged': df, 'meta': {...} } }
    """
    results: Dict[str, Dict[str, Any]] = {}
    for f in uploads:
        try:
            f.seek(0)
        except Exception:
            pass
        name = getattr(f, "name", "uploaded.csv")
        df = pd.read_csv(f)
        out = compute_feature_importance_from_df(
            df, target=target, random_state=random_state, max_missing_ratio=max_missing_ratio
        )
        results[name] = {
            "rf": out["rf"],
            "lr": out["lr"],
            "merged": out["merged"],
            "meta": {
                "kept_columns_after_missing_drop": out["kept_columns_after_missing_drop"],
                "n_rows": int(df.shape[0]),
                "n_cols": int(df.shape[1]),
            },
        }
    return results
