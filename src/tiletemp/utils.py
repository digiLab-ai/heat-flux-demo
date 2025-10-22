# src/tiletemp/utils.py
import numpy as np
import pandas as pd
from typing import Dict, List

NX_FIXED = 50
NZ_FIXED = 40
X_MIN_MM, X_MAX_MM = -50.0, 150.0
Z_MIN_MM, Z_MAX_MM = 0.0, 20.0

CONTROL_KEYS = [
    "P_SOL_MW","flux_expansion","angle_deg",
    "coolant_T_K","k_W_mK","thickness_mm",
    "neutral_fraction","impurity_fraction","ne_19"
]

def fixed_grids():
    x = np.linspace(X_MIN_MM, X_MAX_MM, NX_FIXED)
    z = np.linspace(Z_MIN_MM, Z_MAX_MM, NZ_FIXED)
    return x, z

def pack_controls(params: Dict) -> Dict:
    out = {}
    for k in CONTROL_KEYS:
        v = params.get(k)
        out[k] = float(v) if isinstance(v,(int,float,np.floating)) else v
    return out

def outputs_to_row_2d(T: np.ndarray, prefix="T") -> Dict:
    flat = T.ravel(order="C")
    return {f"{prefix}_{i}": float(flat[i]) for i in range(flat.size)}

def lhs(n_samples: int, n_dims: int, rng: np.random.Generator) -> np.ndarray:
    seg = np.linspace(0, 1, n_samples+1)
    pts = (seg[:-1] + seg[1:]) / 2.0
    X = np.zeros((n_samples, n_dims))
    for d in range(n_dims):
        X[:, d] = rng.permutation(pts)
    return X

def halton(n_samples: int, n_dims: int, rng: np.random.Generator) -> np.ndarray:
    def vdc(n, base):
        v, denom = 0.0, 1.0
        while n:
            n, r = divmod(n, base)
            denom *= base
            v += r / denom
        return v
    primes = [2,3,5,7,11,13,17,19,23,29]
    bases = primes[:n_dims]
    offset = rng.integers(0, 10000)
    H = np.zeros((n_samples, n_dims))
    for i in range(n_samples):
        for d, b in enumerate(bases):
            H[i, d] = vdc(i+1+offset, b)
    return H

def parse_inputs_csv(file_bytes) -> Dict:
    try:
        df = pd.read_csv(file_bytes)
        present = [k for k in CONTROL_KEYS if k in df.columns]
        if len(present) >= 1:
            row = df.iloc[0]
            out = {}
            for k in CONTROL_KEYS:
                if k in df.columns:
                    out[k] = float(row[k])
            return out
        file_bytes.seek(0)
        arr = pd.read_csv(file_bytes, header=None).values.ravel()
        return {k: float(v) for k, v in zip(CONTROL_KEYS, arr)}
    except Exception:
        file_bytes.seek(0)
        arr = np.loadtxt(file_bytes, delimiter=",")
        if arr.ndim > 1:
            arr = arr.ravel()
        return {k: float(v) for k, v in zip(CONTROL_KEYS, arr)}
