# demos-tiletemp-2d — v2

**2D tile temperature (x–z) under an Eich heat-flux boundary**, with fixed grids and added physics:
- Strike offset **fixed at 0**
- **x-range** = [-50, 150] mm with **Nx=50**
- **z-range** = [0, 20] mm with **Nz=40**
- **Neutral pressure** broadens spread (λq & S)
- **Impurity fraction** removes power to tile: P_eff = P_SOL · (1 − impurity)

Also includes:
- Auto-generation of dataset rows via **Latin Hypercube** or **Halton** sampling
- **Compare prediction** tab for drag-and-drop predictions (and optional uncertainties), with RMSE/MAE & plots

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset format
- `inputs.csv` — one row per configuration (controls + fixed-grid metadata)
- `outputs.csv` — flattened 2D field (`T_0 ... T_{Nx*Nz-1}`), row-major (x fastest)

## Notes
- Normalisation ensures toroidal integral of \(q(x)\) ≈ `P_SOL` (plus a small fixed background).
- Conduction is 1D in-depth (teaching simplification). LHS is a good default; Halton provides low-discrepancy coverage.
