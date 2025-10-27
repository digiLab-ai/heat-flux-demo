# src/tiletemp/utils.py
import numpy as np
import pandas as pd
from typing import Dict

NX_FIXED = 30
NZ_FIXED = 20
X_MIN_MM, X_MAX_MM = -50.0, 150.0
Z_MIN_MM, Z_MAX_MM = 0.0, 20.0

CONTROL_KEYS = ["P_SOL_MW", "neutral_fraction", "impurity_fraction", "ne_19"]

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

def parse_inputs_csv_with_header(file_bytes) -> Dict:
    df = pd.read_csv(file_bytes)  # header row expected
    row = df.iloc[0]
    out = {}
    for k in CONTROL_KEYS:
        if k in df.columns:
            out[k] = float(row[k])
    return out

def read_flat_with_header(file_bytes) -> np.ndarray:
    df = pd.read_csv(file_bytes)  # header expected
    arr = df.values.ravel()
    return arr
