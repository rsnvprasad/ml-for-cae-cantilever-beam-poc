from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_builder import build_feature_row
from src.model_loader import load_all_models
from src.physics import calculate_physics_outputs


def calculate_error(actual: float, predicted: float) -> dict:
    """Calculate absolute and percentage error."""
    absolute_error = abs(actual - predicted)

    if abs(actual) > 1e-12:
        percentage_error = (absolute_error / abs(actual)) * 100.0
    else:
        percentage_error = np.nan

    return {
        "absolute_error": absolute_error,
        "percentage_error": percentage_error,
    }


def predict_single_case(
    L_mm: float,
    b_mm: float,
    h_mm: float,
    E_MPa: float,
    F_N: float,
) -> pd.DataFrame:
    """Run physics and ML prediction for one input case."""

    models = load_all_models()

    X = build_feature_row(
        L_mm=L_mm,
        b_mm=b_mm,
        h_mm=h_mm,
        E_MPa=E_MPa,
        F_N=F_N,
    )

    physics = calculate_physics_outputs(
        L_mm=L_mm,
        b_mm=b_mm,
        h_mm=h_mm,
        E_MPa=E_MPa,
        F_N=F_N,
    )

    physics_deflection = physics["deflection_mm"]
    physics_stress = physics["stress_MPa"]

    results = []

    results.append({
        "Model": "Ground Truth",
        "Deflection (mm)": physics_deflection,
        "Stress (MPa)": physics_stress,
        "Deflection Error (mm)": 0.0,
        "Deflection Error (%)": 0.0,
        "Stress Error (MPa)": 0.0,
        "Stress Error (%)": 0.0,
    })

    rf_deflection = float(models["RF"]["deflection"].predict(X)[0])
    rf_stress = float(models["RF"]["stress"].predict(X)[0])

    deflection_error = calculate_error(physics_deflection, rf_deflection)
    stress_error = calculate_error(physics_stress, rf_stress)

    results.append({
        "Model": "Random Forest",
        "Deflection (mm)": rf_deflection,
        "Stress (MPa)": rf_stress,
        "Deflection Error (mm)": deflection_error["absolute_error"],
        "Deflection Error (%)": deflection_error["percentage_error"],
        "Stress Error (MPa)": stress_error["absolute_error"],
        "Stress Error (%)": stress_error["percentage_error"],
    })

    xgb_deflection = float(models["XGB"]["deflection"].predict(X)[0])
    xgb_stress = float(models["XGB"]["stress"].predict(X)[0])

    deflection_error = calculate_error(physics_deflection, xgb_deflection)
    stress_error = calculate_error(physics_stress, xgb_stress)

    results.append({
        "Model": "XGBoost",
        "Deflection (mm)": xgb_deflection,
        "Stress (MPa)": xgb_stress,
        "Deflection Error (mm)": deflection_error["absolute_error"],
        "Deflection Error (%)": deflection_error["percentage_error"],
        "Stress Error (MPa)": stress_error["absolute_error"],
        "Stress Error (%)": stress_error["percentage_error"],
    })

    nn = models["NN_v3_Compact"]

    X_deflection_scaled = nn["deflection_feature_scaler"].transform(
        X[nn["deflection_feature_columns"]]
    )
    nn_deflection_scaled = nn["deflection_model"].predict(
        X_deflection_scaled,
        verbose=0
    )
    nn_deflection = float(
        nn["deflection_target_scaler"]
        .inverse_transform(nn_deflection_scaled.reshape(-1, 1))
        .ravel()[0]
    )

    X_stress_scaled = nn["stress_feature_scaler"].transform(
        X[nn["stress_feature_columns"]]
    )
    nn_stress_scaled = nn["stress_model"].predict(
        X_stress_scaled,
        verbose=0
    )
    nn_stress = float(
        nn["stress_target_scaler"]
        .inverse_transform(nn_stress_scaled.reshape(-1, 1))
        .ravel()[0]
    )

    deflection_error = calculate_error(physics_deflection, nn_deflection)
    stress_error = calculate_error(physics_stress, nn_stress)

    results.append({
        "Model": "Neural Networks",
        "Deflection (mm)": nn_deflection,
        "Stress (MPa)": nn_stress,
        "Deflection Error (mm)": deflection_error["absolute_error"],
        "Deflection Error (%)": deflection_error["percentage_error"],
        "Stress Error (MPa)": stress_error["absolute_error"],
        "Stress Error (%)": stress_error["percentage_error"],
    })

    rbf_deflection_package = models["RBF_Cubic"]["deflection"]
    rbf_stress_package = models["RBF_Cubic"]["stress"]

    rbf_deflection_features = rbf_deflection_package["feature_columns"]
    rbf_stress_features = rbf_stress_package["feature_columns"]

    X_rbf_deflection_scaled = rbf_deflection_package["feature_scaler"].transform(
        X[rbf_deflection_features]
    )
    rbf_deflection = float(
        rbf_deflection_package["rbf_model"](X_rbf_deflection_scaled)[0]
    )

    X_rbf_stress_scaled = rbf_stress_package["feature_scaler"].transform(
        X[rbf_stress_features]
    )
    rbf_stress = float(
        rbf_stress_package["rbf_model"](X_rbf_stress_scaled)[0]
    )

    deflection_error = calculate_error(physics_deflection, rbf_deflection)
    stress_error = calculate_error(physics_stress, rbf_stress)

    results.append({
        "Model": "RBF Interpolator",
        "Deflection (mm)": rbf_deflection,
        "Stress (MPa)": rbf_stress,
        "Deflection Error (mm)": deflection_error["absolute_error"],
        "Deflection Error (%)": deflection_error["percentage_error"],
        "Stress Error (MPa)": stress_error["absolute_error"],
        "Stress Error (%)": stress_error["percentage_error"],
    })

    results_df = pd.DataFrame(results)

    # # Combined ranking score
    # results_df["Combined Error Score"] = (
    #     results_df["Deflection Error (%)"]
    #     + results_df["Stress Error (%)"]
    # )

    # Keep Ground Truth at top
    physics_row = results_df[
        results_df["Model"] == "Ground Truth"
    ]

    ml_rows = results_df[
        results_df["Model"] != "Ground Truth"
    ]

    # ml_rows = ml_rows.sort_values(
    #     by="Combined Error Score"
    # )

    results_df = pd.concat(
        [physics_row, ml_rows],
        ignore_index=True
    )

    return results_df


def test_prediction() -> None:
    """Quick local test."""

    results_df = predict_single_case(
        L_mm=620,
        b_mm=54,
        h_mm=24,
        E_MPa=165000,
        F_N=1100,
    )

    print("\nPrediction Comparison:")
    print(results_df.round(4).to_string(index=False))


if __name__ == "__main__":
    test_prediction()