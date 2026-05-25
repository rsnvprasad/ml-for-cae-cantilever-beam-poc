from __future__ import annotations


def calculate_moment_of_inertia(b_mm: float, h_mm: float) -> float:
    """Calculate rectangular section moment of inertia in mm^4."""
    return (b_mm * h_mm**3) / 12.0


def calculate_deflection_mm(
    L_mm: float,
    b_mm: float,
    h_mm: float,
    E_MPa: float,
    F_N: float,
) -> float:
    """
    Cantilever beam tip deflection.

    Formula:
        delta = F * L^3 / (3 * E * I)

    Units:
        F = N
        L = mm
        E = MPa = N/mm^2
        I = mm^4
        delta = mm
    """
    I_mm4 = calculate_moment_of_inertia(b_mm, h_mm)
    return (F_N * L_mm**3) / (3.0 * E_MPa * I_mm4)


def calculate_stress_MPa(
    L_mm: float,
    b_mm: float,
    h_mm: float,
    F_N: float,
) -> float:
    """
    Cantilever beam maximum bending stress.

    Formula:
        sigma = M*c/I = 6*F*L/(b*h^2)

    Units:
        F = N
        L = mm
        b,h = mm
        stress = MPa
    """
    return (6.0 * F_N * L_mm) / (b_mm * h_mm**2)


def calculate_physics_outputs(
    L_mm: float,
    b_mm: float,
    h_mm: float,
    E_MPa: float,
    F_N: float,
) -> dict:
    """Return physics-based deflection and stress."""
    return {
        "I_mm4": calculate_moment_of_inertia(b_mm, h_mm),
        "deflection_mm": calculate_deflection_mm(L_mm, b_mm, h_mm, E_MPa, F_N),
        "stress_MPa": calculate_stress_MPa(L_mm, b_mm, h_mm, F_N),
    }