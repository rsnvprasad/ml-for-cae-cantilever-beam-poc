from __future__ import annotations

from pathlib import Path
import json
import joblib


PROJECT_ROOT = Path(__file__).resolve().parent.parent

APP_MODELS_DIR = PROJECT_ROOT / "app_models"


MODEL_PATHS = {
    "rf_deflection": APP_MODELS_DIR / "random_forest" / "deflection_mm_rf_v2_tuned.pkl",
    "rf_stress": APP_MODELS_DIR / "random_forest" / "stress_MPa_rf_v2_tuned.pkl",

    "xgb_deflection": APP_MODELS_DIR / "xgboost" / "deflection_mm_xgb_v1.pkl",
    "xgb_stress": APP_MODELS_DIR / "xgboost" / "stress_MPa_xgb_v1.pkl",

    "rbf_deflection": APP_MODELS_DIR / "rbf_interpolator" / "deflection_mm" / "rbf_cubic_model_package.pkl",
    "rbf_stress": APP_MODELS_DIR / "rbf_interpolator" / "stress_MPa" / "rbf_cubic_model_package.pkl",

    "nn_deflection_model": APP_MODELS_DIR / "neural_network" / "deflection_mm" / "nn_v3_compact_model.keras",
    "nn_deflection_feature_scaler": APP_MODELS_DIR / "neural_network" / "deflection_mm" / "nn_v3_compact_feature_scaler.pkl",
    "nn_deflection_target_scaler": APP_MODELS_DIR / "neural_network" / "deflection_mm" / "nn_v3_compact_target_scaler.pkl",
    "nn_deflection_feature_columns": APP_MODELS_DIR / "neural_network" / "deflection_mm" / "nn_v3_compact_feature_columns.json",

    "nn_stress_model": APP_MODELS_DIR / "neural_network" / "stress_MPa" / "nn_v3_compact_model.keras",
    "nn_stress_feature_scaler": APP_MODELS_DIR / "neural_network" / "stress_MPa" / "nn_v3_compact_feature_scaler.pkl",
    "nn_stress_target_scaler": APP_MODELS_DIR / "neural_network" / "stress_MPa" / "nn_v3_compact_target_scaler.pkl",
    "nn_stress_feature_columns": APP_MODELS_DIR / "neural_network" / "stress_MPa" / "nn_v3_compact_feature_columns.json",
}


def load_joblib_model(path: Path):
    """Load a joblib/pickle model file."""
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def load_json(path: Path) -> dict | list:
    """Load JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_keras_model(path: Path):
    """Load a Keras model only when needed."""
    if not path.exists():
        raise FileNotFoundError(f"Keras model file not found: {path}")

    from tensorflow.keras.models import load_model

    return load_model(path)


def load_all_models() -> dict:
    """Load all final app models."""

    models = {
        "RF": {
            "deflection": load_joblib_model(MODEL_PATHS["rf_deflection"]),
            "stress": load_joblib_model(MODEL_PATHS["rf_stress"]),
        },
        "XGB": {
            "deflection": load_joblib_model(MODEL_PATHS["xgb_deflection"]),
            "stress": load_joblib_model(MODEL_PATHS["xgb_stress"]),
        },
        "RBF_Cubic": {
            "deflection": load_joblib_model(MODEL_PATHS["rbf_deflection"]),
            "stress": load_joblib_model(MODEL_PATHS["rbf_stress"]),
        },
        "NN_v3_Compact": {
            "deflection_model": load_keras_model(MODEL_PATHS["nn_deflection_model"]),
            "deflection_feature_scaler": load_joblib_model(MODEL_PATHS["nn_deflection_feature_scaler"]),
            "deflection_target_scaler": load_joblib_model(MODEL_PATHS["nn_deflection_target_scaler"]),
            "deflection_feature_columns": load_json(MODEL_PATHS["nn_deflection_feature_columns"]),

            "stress_model": load_keras_model(MODEL_PATHS["nn_stress_model"]),
            "stress_feature_scaler": load_joblib_model(MODEL_PATHS["nn_stress_feature_scaler"]),
            "stress_target_scaler": load_joblib_model(MODEL_PATHS["nn_stress_target_scaler"]),
            "stress_feature_columns": load_json(MODEL_PATHS["nn_stress_feature_columns"]),
        },
    }

    return models