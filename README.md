# demos-tiletemp-2d

**2D tile temperature (x–z) under an Eich heat-flux boundary.**  
Surface receives a 1D Eich profile \(q(x)\); temperature through the tile is computed from steady conduction with back-face coolant.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Controls (operator/engineering relevant)
- `P_SOL` [MW], `flux_expansion` [-], `angle_deg` [deg], `x0_mm` [mm]
- `coolant_T_K` [K], `k_W_mK` [W/m·K], `thickness_mm` [mm]
- Grid: `x_min/max`, `nx`, `z_min/max`, `nz`

## Output
- Heatmap **T(x,z)** [K]
- Dataset export: `inputs.csv` (one row per config) and `outputs.csv` (flattened `T_0..T_{nx*nz-1}` row-major).

## Notes
- Normalisation ensures toroidal integral of \(q(x)\) ≈ `P_SOL` (plus a small fixed background).
- Conduction is 1D in-depth (no lateral conduction) to keep the model teachable.
- Great for multi-physics surrogate demos (plasma → materials thermal response).
