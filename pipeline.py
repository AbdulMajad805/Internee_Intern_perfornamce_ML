import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import shap
import warnings
import pickle
import os

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Constants ──────────────────────────────────────────────
FEATURE_COLS = [
    "task_completion_rate",
    "avg_task_completion_time",
    "attendance_percentage",
    "feedback_rating",
    "deadlines_missed",
    "github_commits",
    "communication_score",
    "learning_speed",
    "meeting_participation",
    "work_hours_per_day",
    "stress_level",
]

ENGINEERED_COLS = [
    "productivity_index",
    "reliability_score",
    "engagement_score",
    "burnout_risk",
    "overall_quality",
]

ALL_FEATURES = FEATURE_COLS + ENGINEERED_COLS

RISK_THRESHOLDS = {
    "critical":  35,
    "at_risk":   45,
    "on_track":  65,
    "excelling": 100,
}

# ── Data Loading ───────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = FEATURE_COLS + ["final_performance_score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df


# ── Feature Engineering ────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Productivity: high completion + low time + low missed deadlines
    df["productivity_index"] = (
        df["task_completion_rate"] / 100 *
        (1 - df["avg_task_completion_time"] / 15) *
        (1 - df["deadlines_missed"] / 10)
    ).clip(0, 1) * 100

    df["reliability_score"] = (
        df["attendance_percentage"] * 0.6 +
        (1 - df["deadlines_missed"] / 10) * 100 * 0.4
    ).clip(0, 100)

    df["engagement_score"] = (
        df["communication_score"] / 10 * 25 +
        df["learning_speed"] / 10 * 25 +
        df["meeting_participation"] / 10 * 25 +
        df["github_commits"] / 300 * 25
    ).clip(0, 100)

    df["burnout_risk"] = (
        df["stress_level"] / 10 * 60 +
        (df["work_hours_per_day"] / 12) * 40
    ).clip(0, 100)

    df["overall_quality"] = (
        df["productivity_index"] * 0.35 +
        df["reliability_score"]  * 0.25 +
        df["engagement_score"]   * 0.25 +
        (100 - df["burnout_risk"]) * 0.15
    ).clip(0, 100)

    return df


# ── Train / Evaluate ───────────────────────────────────────

def train_models(df: pd.DataFrame):
    df_eng = engineer_features(df)
    X = df_eng[ALL_FEATURES]
    y = df_eng["final_performance_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}
    models  = {}

    # ── Random Forest with tuning
    rf_params = {
        "n_estimators":    [100, 200, 300],
        "max_depth":       [6, 8, 10, None],
        "min_samples_leaf":[2, 3, 5],
        "max_features":    ["sqrt", "log2"],
    }
    rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_search = RandomizedSearchCV(
        rf_base, rf_params, n_iter=20, cv=3,
        scoring="r2", random_state=42, n_jobs=-1
    )
    rf_search.fit(X_train, y_train)
    rf = rf_search.best_estimator_
    rf_pred = rf.predict(X_test)
    cv_rf   = cross_val_score(rf, X_train, y_train, cv=5, scoring="r2")

    results["Random Forest"] = {
        "mae":       round(mean_absolute_error(y_test, rf_pred), 3),
        "rmse":      round(np.sqrt(mean_squared_error(y_test, rf_pred)), 3),
        "r2":        round(r2_score(y_test, rf_pred), 4),
        "cv_r2":     round(cv_rf.mean(), 4),
        "cv_std":    round(cv_rf.std(), 4),
        "best_params": rf_search.best_params_,
    }
    models["Random Forest"] = rf

    # ── XGBoost with tuning
    xgb_params = {
        "n_estimators":     [200, 300, 400],
        "max_depth":        [4, 5, 6],
        "learning_rate":    [0.03, 0.05, 0.1],
        "subsample":        [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "reg_alpha":        [0, 0.1, 0.5],
        "reg_lambda":       [1, 1.5, 2],
    }
    xgb_base = xgb.XGBRegressor(random_state=42, verbosity=0)
    xgb_search = RandomizedSearchCV(
        xgb_base, xgb_params, n_iter=20, cv=3,
        scoring="r2", random_state=42, n_jobs=-1
    )
    xgb_search.fit(X_train, y_train)
    xgb_model = xgb_search.best_estimator_
    xgb_pred  = xgb_model.predict(X_test)
    cv_xgb    = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring="r2")

    results["XGBoost"] = {
        "mae":       round(mean_absolute_error(y_test, xgb_pred), 3),
        "rmse":      round(np.sqrt(mean_squared_error(y_test, xgb_pred)), 3),
        "r2":        round(r2_score(y_test, xgb_pred), 4),
        "cv_r2":     round(cv_xgb.mean(), 4),
        "cv_std":    round(cv_xgb.std(), 4),
        "best_params": xgb_search.best_params_,
    }
    models["XGBoost"] = xgb_model

    ens_pred = (rf_pred + xgb_pred) / 2
    results["Ensemble"] = {
        "mae":    round(mean_absolute_error(y_test, ens_pred), 3),
        "rmse":   round(np.sqrt(mean_squared_error(y_test, ens_pred)), 3),
        "r2":     round(r2_score(y_test, ens_pred), 4),
        "cv_r2":  None,
        "cv_std": None,
        "best_params": {},
    }

    best_name  = max(["Random Forest", "XGBoost"], key=lambda n: results[n]["r2"])
    best_model = models[best_name]

    # Feature importance
    fi = pd.Series(
        best_model.feature_importances_, index=ALL_FEATURES
    ).sort_values(ascending=False)

    return models, best_model, best_name, results, fi, X_test, y_test, X_train, y_train


def compute_shap(model, X_sample: pd.DataFrame, model_name: str):
    """Return SHAP values and explainer for a sample of rows."""
    if model_name == "XGBoost":
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values


def assign_risk(score: float) -> str:
    if score < RISK_THRESHOLDS["critical"]:
        return "Critical"
    elif score < RISK_THRESHOLDS["at_risk"]:
        return "At Risk"
    elif score < RISK_THRESHOLDS["on_track"]:
        return "On Track"
    else:
        return "Excelling"


def flag_interns(df: pd.DataFrame, model, best_name: str) -> pd.DataFrame:
    df_eng  = engineer_features(df)
    X       = df_eng[ALL_FEATURES]
    preds   = model.predict(X)

    out = df[["intern_id", "intern_name"] + FEATURE_COLS].copy()
    out["predicted_score"]    = preds.round(1)
    out["actual_score"]       = df["final_performance_score"].values
    out["performance_category"] = df["performance_category"].values
    out["risk_tier"]          = [assign_risk(s) for s in preds]
    out["score_gap"]          = (out["actual_score"] - out["predicted_score"]).round(1)

    def alert_reasons(row):
        reasons = []
        if row["deadlines_missed"] >= 7:        reasons.append("High missed deadlines")
        if row["stress_level"] >= 8:            reasons.append("High stress level")
        if row["attendance_percentage"] <= 60:  reasons.append("Low attendance")
        if row["task_completion_rate"] <= 50:   reasons.append("Low task completion")
        if row["feedback_rating"] <= 2.0:       reasons.append("Poor feedback rating")
        if row["avg_task_completion_time"] >= 12: reasons.append("Slow task completion")
        return " | ".join(reasons) if reasons else "—"

    out["alert_reasons"] = out.apply(alert_reasons, axis=1)
    return out.sort_values("predicted_score")


def predict_single(model, feature_dict: dict) -> dict:
    row = pd.DataFrame([feature_dict])
    row_eng = engineer_features(row)
    X   = row_eng[ALL_FEATURES]
    score = float(model.predict(X)[0])
    score = round(np.clip(score, 0, 100), 1)
    return {
        "score":    score,
        "risk_tier": assign_risk(score),
        "engineered": {
            "productivity_index": round(float(row_eng["productivity_index"].iloc[0]), 1),
            "reliability_score":  round(float(row_eng["reliability_score"].iloc[0]), 1),
            "engagement_score":   round(float(row_eng["engagement_score"].iloc[0]), 1),
            "burnout_risk":       round(float(row_eng["burnout_risk"].iloc[0]), 1),
            "overall_quality":    round(float(row_eng["overall_quality"].iloc[0]), 1),
        }
    }

def save_model(model, path="model.pkl"):
    with open(path, "wb") as f:
        pickle.dump(model, f)

def load_model(path="model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)