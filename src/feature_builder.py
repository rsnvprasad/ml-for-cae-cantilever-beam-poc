from __future__ import annotations

import pandas as pd


FEATURE_COLUMNS = [
    "L_mm", "b_mm", "h_mm", "E_MPa", "F_N",
    "L_cu", "h_sq", "h_cu",
    "b_h2", "b_h3",
    "I_calc", "inv_E", "inv_I",
    "F_L", "F_L3", "F_over_E",
    "slenderness",
]


def build_feature_row(
    L_mm: float,
    b_mm: float,
    h_mm: float,
    E_MPa: float,
    F_N: float,
) -> pd.DataFrame:
    """Build one model-ready feature row using the same feature names/order used during training."""

    I_calc = (b_mm * h_mm**3) / 12.0

    data = {
        "L_mm": L_mm,
        "b_mm": b_mm,
        "h_mm": h_mm,
        "E_MPa": E_MPa,
        "F_N": F_N,
        "L_cu": L_mm**3,
        "h_sq": h_mm**2,
        "h_cu": h_mm**3,
        "b_h2": b_mm * h_mm**2,
        "b_h3": b_mm * h_mm**3,
        "I_calc": I_calc,
        "inv_E": 1.0 / E_MPa,
        "inv_I": 1.0 / I_calc,
        "F_L": F_N * L_mm,
        "F_L3": F_N * L_mm**3,
        "F_over_E": F_N / E_MPa,
        "slenderness": L_mm / h_mm,
    }

    return pd.DataFrame([data], columns=FEATURE_COLUMNS)