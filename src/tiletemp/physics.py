# src/tiletemp/physics.py
import numpy as np

def erfc(x: np.ndarray) -> np.ndarray:
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
    lambda_q_mm_base: float = 5.0,
    S_mm_base: float = 1.0,
    q_bg_MW_m2: float = 0.02,
    toroidal_radius_m: float = 1.0,
    neutral_fraction: float = 0.0,
    neutral_alpha: float = 0.20,
    ne_19: float = 5.0,
    ne_ref: float = 5.0,
    ne_alpha: float = 0.08,
) -> np.ndarray:
    x = np.asarray(x_mm)
    angle_rad = np.deg2rad(max(angle_deg, 0.5))
    f_angle = (0.5 / angle_rad)
    f_flux  = (flux_expansion / 5.0)
    f_neut  = (1.0 + neutral_alpha * max(neutral_fraction, 0.0))
    f_ne    = (1.0 + ne_alpha * (max(ne_19, 0.0) - ne_ref))
    lam = max(lambda_q_mm_base * f_angle * f_flux * f_neut * f_ne, 0.2)
    S   = max(S_mm_base         * f_angle * f_flux * f_neut * np.sqrt(f_ne), 0.1)

    xi = x
    A = (S / (2.0 * lam)) ** 2 - (xi / lam)
    B = (S / (2.0 * lam)) - (xi / S)
    shape = 0.5 * np.exp(A) * erfc(B)

    dx_m = (x[1]-x[0]) * 1e-3 if x.size>1 else 1e-3
    circum_m = 2.0*np.pi*toroidal_radius_m
    base_integral = float(np.sum(shape) * dx_m * circum_m)
    q0 = 0.0 if base_integral<=0 else (P_SOL_MW / max(base_integral, 1e-12))

    q = q0*shape + q_bg_MW_m2
    return np.clip(q, 0.0, None)

def tile_temperature_field(
    x_mm: np.ndarray,
    z_mm: np.ndarray,
    P_SOL_MW: float,
    flux_expansion: float,
    angle_deg: float,
    coolant_T_K: float,
    k_W_mK: float,
    thickness_mm: float,
    lambda_q_mm_base: float = 5.0,
    S_mm_base: float = 1.0,
    q_bg_MW_m2: float = 0.02,
    toroidal_radius_m: float = 1.0,
    neutral_fraction: float = 0.0,
    impurity_fraction: float = 0.0,
    ne_19: float = 5.0,
) -> np.ndarray:
    P_eff = P_SOL_MW * (1.0 - max(min(impurity_fraction, 0.95), 0.0))
    qx_MW_m2 = eich_1d_profile(
        x_mm,
        P_SOL_MW=P_eff,
        flux_expansion=flux_expansion,
        angle_deg=angle_deg,
        lambda_q_mm_base=lambda_q_mm_base,
        S_mm_base=S_mm_base,
        q_bg_MW_m2=q_bg_MW_m2,
        toroidal_radius_m=toroidal_radius_m,
        neutral_fraction=neutral_fraction,
        ne_19=ne_19,
    )
    qx_W_m2 = qx_MW_m2 * 1e6
    z_m = np.asarray(z_mm) * 1e-3
    thickness_m = float(thickness_mm) * 1e-3
    z_m = np.clip(z_m, 0.0, thickness_m)
    T = coolant_T_K + np.outer(qx_W_m2 / max(k_W_mK, 1e-6), (thickness_m - z_m))
    return T
