# src/tiletemp/utils.py
import numpy as np
import pandas as pd
from typing import Dict

def default_grids(x_min_mm=-100.0, x_max_mm=100.0, nx=300, z_min_mm=0.0, z_max_mm=20.0, nz=120):
    x = np.linspace(x_min_mm, x_max_mm, nx)
    z = np.linspace(z_min_mm, z_max_mm, nz)
    return x, z

def pack_controls(params: Dict) -> Dict:
    keys = [
        "P_SOL_MW","flux_expansion","angle_deg","x0_mm",
        "coolant_T_K","k_W_mK","thickness_mm",
        "x_min_mm","x_max_mm","nx","z_min_mm","z_max_mm","nz"
    ]
    out = {}
    for k in keys:
        v = params.get(k)
        out[k] = float(v) if isinstance(v,(int,float,float)) else v
    return out

def outputs_to_row_2d(T: np.ndarray, prefix="T") -> Dict:
    flat = T.ravel(order="C")
    return {f"{prefix}_{i}": float(flat[i]) for i in range(flat.size)}
