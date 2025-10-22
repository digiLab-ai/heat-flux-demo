# demos-tiletemp-2d — v4

**2D tile temperature (x–z) under an Eich heat-flux boundary** with fixed smaller grids and simplified sampling.

### v4 changes
- All CSVs are assumed to have a **header row**.
- Grid reduced to **Nx=30**, **Nz=20** (x: \[-50, 150] mm; z: \[0, 20] mm).
- Auto-sampling uses **Latin Hypercube only** (default **N=20**).
- **Filename fields** added for dataset downloads (inputs/outputs).

### Controls (saved to inputs.csv)
`P_SOL_MW, flux_expansion, angle_deg, coolant_T_K, k_W_mK, thickness_mm, neutral_fraction, impurity_fraction, ne_19`

### Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Compare tab
- Upload **Inputs CSV** (with headers) to set the ground-truth configuration.
- Upload **Prediction CSV** (flattened, length 30×20=600) and **Uncertainty CSV** (same length, 1σ); both must have headers.
- Visuals: **plasma** colormap for T, **seismic** symmetric for error, **Reds** for 95% CI (1.96×σ).
- Metrics (RMSE, MAE, R², MSLL) shown under an expander.

### Dataset format
- `inputs.csv` — one row per configuration (only controls)
- `outputs.csv` — flattened `T_0 ... T_{Nx*Nz-1}` (row-major, x-fastest)
