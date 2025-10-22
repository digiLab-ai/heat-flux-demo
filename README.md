# demos-tiletemp-2d — v3

**2D tile temperature (x–z) under an Eich heat-flux boundary** with improved compare tools, colors, and physics.

### v3 highlights
- **CSV uploads**: headers in prediction/uncertainty files are ignored. Inputs CSV sets ground truth.
- **Colormaps**: Temperature `'plasma'`; Error `'seismic'` symmetric about 0; Uncertainty `'Reds'` shown as **95% CI = 1.96×σ**.
- **Compare**: Requires **both** prediction and uncertainty; 2×2 grid (GT, Pred, Error, Uncertainty). Metrics (RMSE/MAE/R²/**MSLL**) under an expander.
- **Saved CSVs**: Only true control parameters are saved (no fixed metadata).
- **Physics**: `neutral_fraction` + `ne_19` broaden the Eich footprint (illustrative).

### Controls
`P_SOL_MW, flux_expansion, angle_deg, coolant_T_K, k_W_mK, thickness_mm, neutral_fraction, impurity_fraction, ne_19`

### Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Dataset format
- `inputs.csv` — one row per configuration (only controls)
- `outputs.csv` — flattened `T_0 ... T_{Nx*Nz-1}` (row-major, x-fastest)
