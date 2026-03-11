"""
baseline_model.py
-----------------
Random Forest regression baseline.

Feature construction (no heavy embeddings — keeps it simple and fast):
  - Numerical (15 features):
      experience_min_years, age_min, age_max, years_since_graduation,
      gpa_normalized, total_work_experience_years, has_certification,
      has_career_objective, num_certifications,
      university_rank_score, university_world_rank, university_is_ranked,
      company_is_fortune100, company_fortune100_rank_norm, company_size_norm
  - Categorical (one-hot via sklearn):
      degree_level, result_type
  - Top-K job position (one-hot, top 50 + 'other'):
      job_position_name
  - Text (TF-IDF, 200 features each):
      career_objective, skills, responsibilities,
      skills_required, educationaL_requirements

All assembled via sklearn Pipeline + ColumnTransformer.

Usage:
    python baseline_model.py
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, f1_score)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

NUMERICAL_COLS = [
    "experience_min_years",
    "age_min",
    "age_max",
    "years_since_graduation",
    "gpa_normalized",
    "total_work_experience_years",
    "has_certification",
    "has_career_objective",
    "num_certifications",
    # University ranking features (CWUR/Shanghai/Times)
    "university_rank_score",
    "university_world_rank",
    "university_is_ranked",
    # Fortune 100 company features (Largest_Companies.csv)
    "company_is_fortune100",
    "company_fortune100_rank_norm",
    "company_size_norm",
]

CATEGORICAL_COLS = ["degree_level", "result_type"]

# Top-50 job positions will be kept; the rest become 'other'
JOB_POSITION_COL = "job_position_name"

TEXT_COLS = [
    "career_objective",
    "skills",
    "responsibilities",
    "skills_required",
    "educationaL_requirements",
]

TARGET = "matched_score"

TFIDF_MAX_FEATURES = 200   # per text column
TOP_K_JOBS = 50
RANDOM_SEED = 42
# Threshold to convert regression output → binary class for Accuracy / F1
# (matched_score ≥ 0.5 → "good match",  < 0.5 → "poor match")
MATCH_THRESHOLD = 0.5


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _cap_job_position(df: pd.DataFrame, top_k: int = TOP_K_JOBS) -> pd.DataFrame:
    """Replace rare job positions with 'other'."""
    df = df.copy()
    top = df[JOB_POSITION_COL].value_counts().nlargest(top_k).index
    df[JOB_POSITION_COL] = df[JOB_POSITION_COL].where(
        df[JOB_POSITION_COL].isin(top), other="other"
    )
    return df


def build_pipeline() -> Pipeline:
    """Construct the full sklearn preprocessing + RF pipeline."""

    # Individual transformers
    num_transformer = StandardScaler()

    cat_transformer = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False
    )

    # One TF-IDF vectorizer per text column
    tfidf_transformers = [
        (f"tfidf_{col}", TfidfVectorizer(max_features=TFIDF_MAX_FEATURES,
                                          ngram_range=(1, 2),
                                          sublinear_tf=True,
                                          min_df=2),
         col)
        for col in TEXT_COLS
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, NUMERICAL_COLS),
            ("cat", cat_transformer, CATEGORICAL_COLS + [JOB_POSITION_COL]),
        ] + tfidf_transformers,
        remainder="drop",
    )

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", rf),
    ])
    return pipeline


# ──────────────────────────────────────────────────────────────────────────────
# Training and evaluation
# ──────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(
    data_path: str,
    model_out_path: str = None,
    test_size: float = 0.2,
    val_size: float = 0.2,
    plot: bool = True,
) -> dict:
    """
    Load cleaned data, train a Random Forest, and return metrics.

    Split: 60% train / 20% val / 20% test  (seed=42)

    Returns
    -------
    dict with keys: train_rmse, val_rmse, test_rmse, train_mae, val_mae,
                    test_mae, train_r2, val_r2, test_r2,
                    train_acc, val_acc, test_acc, train_f1, val_f1, test_f1
    """
    print("Loading data ...")
    df = pd.read_csv(data_path)
    df = _cap_job_position(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET].values

    # Fill any residual NaN in text columns with empty string
    for col in TEXT_COLS:
        X[col] = X[col].fillna("").astype(str)

    # Split: 60% train / 20% val / 20% test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED
    )
    relative_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=relative_val, random_state=RANDOM_SEED
    )

    print(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    pipeline = build_pipeline()

    print("Training Random Forest ...")
    pipeline.fit(X_train, y_train)

    def _metrics(name, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)
        # Binary classification metrics at MATCH_THRESHOLD
        y_true_bin = (y_true  >= MATCH_THRESHOLD).astype(int)
        y_pred_bin = (y_pred  >= MATCH_THRESHOLD).astype(int)
        acc = accuracy_score(y_true_bin, y_pred_bin)
        f1  = f1_score(y_true_bin, y_pred_bin, average="weighted", zero_division=0)
        print(f"  [{name}]  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}"
              f"  Acc={acc:.4f}  F1(wtd)={f1:.4f}")
        return rmse, mae, r2, acc, f1

    pred_train = pipeline.predict(X_train)
    pred_val   = pipeline.predict(X_val)
    pred_test  = pipeline.predict(X_test)

    print(f"\nResults  (classification threshold = {MATCH_THRESHOLD}):")
    tr_rmse, tr_mae, tr_r2, tr_acc, tr_f1 = _metrics("Train", y_train, pred_train)
    vl_rmse, vl_mae, vl_r2, vl_acc, vl_f1 = _metrics("Val  ", y_val,   pred_val)
    te_rmse, te_mae, te_r2, te_acc, te_f1 = _metrics("Test ", y_test,  pred_test)

    # ── Feature importance ──────────────────────────────────────────────────
    rf_model = pipeline.named_steps["model"]
    prep = pipeline.named_steps["preprocessor"]
    feat_names = prep.get_feature_names_out()
    importances = rf_model.feature_importances_

    top_idx = np.argsort(importances)[::-1][:20]
    top_names = [feat_names[i] for i in top_idx]
    top_vals  = importances[top_idx]

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Prediction scatter
        ax = axes[0]
        ax.scatter(y_test, pred_test, alpha=0.2, s=10, color='steelblue')
        lo, hi = min(y_test.min(), pred_test.min()), max(y_test.max(), pred_test.max())
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1)
        ax.set_xlabel("True matched_score")
        ax.set_ylabel("Predicted matched_score")
        ax.set_title(f"Test predictions  (RMSE={te_rmse:.4f}, R²={te_r2:.4f})")
        ax.grid(True, alpha=0.3)

        # Feature importance
        ax = axes[1]
        ax.barh(top_names[::-1], top_vals[::-1], color='teal')
        ax.set_xlabel("Importance")
        ax.set_title("Top-20 feature importances")
        ax.tick_params(axis='y', labelsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(data_path), "baseline_results.png"),
                    dpi=150, bbox_inches='tight')
        plt.show()
        print("Plot saved to data/baseline_results.png")

    # ── Save model ──────────────────────────────────────────────────────────
    if model_out_path is None:
        model_out_path = os.path.join(
            os.path.dirname(data_path), "baseline_rf.joblib"
        )
    joblib.dump(pipeline, model_out_path)
    print(f"\nModel saved to {model_out_path}")

    return dict(
        train_rmse=tr_rmse, val_rmse=vl_rmse, test_rmse=te_rmse,
        train_mae=tr_mae,   val_mae=vl_mae,   test_mae=te_mae,
        train_r2=tr_r2,     val_r2=vl_r2,     test_r2=te_r2,
        train_acc=tr_acc,   val_acc=vl_acc,   test_acc=te_acc,
        train_f1=tr_f1,     val_f1=vl_f1,     test_f1=te_f1,
        top_feature_names=top_names,
        top_feature_importances=top_vals,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results = train_and_evaluate(
        data_path=os.path.join(base, "data", "cleaned_resume_data.csv"),
        plot=True,
    )
