# src/tiletemp/physics.py
import numpy as np

def erfc(x: np.ndarray) -> np.ndarray:
    # Abramowitz & Stegun approximation (sufficient for workshop)
    sign = np.sign(x)
    xa = np.abs(x)
    t = 1.0 / (1.0 + 0.5 * xa)
    tau = t * np.exp(
        -xa*xa
        - 1.26551223
        + 1.00002368 * t
        + 0.37409196 * t**2
        + 0.09678418 * t**3
        - 0.18628806 * t**4
        + 0.27886807 * t**5
        - 1.13520398 * t**6
        + 1.48851587 * t**7
        - 0.82215223 * t**8
        + 0.17087277 * t**9
    )
    erf_approx = 1.0 - tau
    return 1.0 - erf_approx * sign

def eich_1d_profile(
    x_mm: np.ndarray,
    P_SOL_MW: float = 5.0,
    flux_expansion: float = 5.0,
    angle_deg: float = 2.0,
    x0_mm: float = 0.0,
    lambda_q_mm_base: float = 5.0,
    S_mm_base: float = 1.0,
    q_bg_MW_m2: float = 0.02,
    toroidal_radius_m: float = 1.0,
) -> np.ndarray:
    """
    1D Eich profile along strike coordinate x (mm), normalised so that
    ∮ (around torus) ∫ q(x) dx = P_SOL (MW) plus background.
    """
    x = np.asarray(x_mm)
    angle_rad = np.deg2rad(max(angle_deg, 0.5))
    lam = max(lambda_q_mm_base * (flux_expansion/5.0) * (0.5/angle_rad), 0.2)
    S = max(S_mm_base * (flux_expansion/5.0) * (0.5/angle_rad), 0.1)

    xi = x - x0_mm
    A = (S / (2.0 * lam)) ** 2 - (xi / lam)
    B = (S / (2.0 * lam)) - (xi / S)
    shape = 0.5 * np.exp(A) * erfc(B)  # unitless positive

    # Normalisation: toroidal integral of (q0*shape) equals P_SOL (MW)
    dx_m = (x[1]-x[0]) * 1e-3 if x.size>1 else 1e-3
    circum_m = 2.0*np.pi*toroidal_radius_m
    base_integral = float(np.sum(shape) * dx_m * circum_m)  # m^2
    q0 = 0.0 if base_integral<=0 else (P_SOL_MW / max(base_integral, 1e-12))  # MW/m^2 factor

    q = q0*shape + q_bg_MW_m2
    return np.clip(q, 0.0, None)

def tile_temperature_field(
    x_mm: np.ndarray,
    z_mm: np.ndarray,
    P_SOL_MW: float,
    flux_expansion: float,
    angle_deg: float,
    x0_mm: float,
    coolant_T_K: float,
    k_W_mK: float,
    thickness_mm: float,
    lambda_q_mm_base: float = 5.0,
    S_mm_base: float = 1.0,
    q_bg_MW_m2: float = 0.02,
    toroidal_radius_m: float = 1.0,
) -> np.ndarray:
    """
    Steady 1D-through-thickness conduction model at each x:
      -k dT/dz |_{z=0} = q(x)
      T(z=thickness) = coolant_T
    => T(x,z) = coolant_T + q(x)/k * (thickness - z)
    Assumes no lateral conduction (separable x,z) — good for teaching.
    """
    qx_MW_m2 = eich_1d_profile(
        x_mm,
        P_SOL_MW=P_SOL_MW,
        flux_expansion=flux_expansion,
        angle_deg=angle_deg,
        x0_mm=x0_mm,
        lambda_q_mm_base=lambda_q_mm_base,
        S_mm_base=S_mm_base,
        q_bg_MW_m2=q_bg_MW_m2,
        toroidal_radius_m=toroidal_radius_m,
    )
    qx_W_m2 = qx_MW_m2 * 1e6  # convert to W/m^2

    z_m = np.asarray(z_mm) * 1e-3
    thickness_m = float(thickness_mm) * 1e-3
    # Ensure z bounds
    z_m = np.clip(z_m, 0.0, thickness_m)

    # Outer form: T(x,z) = Tc + (qx/k) * (thickness - z)
    T = coolant_T_K + np.outer(qx_W_m2 / max(k_W_mK, 1e-6), (thickness_m - z_m))
    return T
