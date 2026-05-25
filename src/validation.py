from __future__ import annotations


TRAINING_RANGES = {
    "L_mm": {"min": 300.0, "max": 1000.0},
    "b_mm": {"min": 30.0, "max": 70.0},
    "h_mm": {"min": 10.0, "max": 100.0},
    "E_MPa": {"min": 70000.0, "max": 210000.0},
    "F_N": {"min": 500.0, "max": 2000.0},
}


def check_input_ranges(
    L_mm: float,
    b_mm: float,
    h_mm: float,
    E_MPa: float,
    F_N: float,
) -> dict:
    """Check whether input values are inside the model training range."""

    inputs = {
        "L_mm": L_mm,
        "b_mm": b_mm,
        "h_mm": h_mm,
        "E_MPa": E_MPa,
        "F_N": F_N,
    }

    results = {}

    for key, value in inputs.items():
        lower = TRAINING_RANGES[key]["min"]
        upper = TRAINING_RANGES[key]["max"]

        results[key] = {
            "value": value,
            "min": lower,
            "max": upper,
            "inside_range": lower <= value <= upper,
        }

    return results


def is_design_inside_training_range(range_results: dict) -> bool:
    """Return True only if all inputs are inside training range."""
    return all(item["inside_range"] for item in range_results.values())