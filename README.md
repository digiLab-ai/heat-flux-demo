# Demo for the tile temperature of a tokamak divertor tile

**Fixed parameters**
- thickness_mm = 20.0
- k_W_mK = 120.0
- coolant_T_K = 350.0
- angle_deg = 2.0
- flux_expansion = 5.0

**Free parameters**: `P_SOL_MW`, `neutral_fraction`, `impurity_fraction`, `ne_19`

- Tile temperature grid 30×20.
- Generate data with tunable free parameters.
- Generate data with Latin Hypercube Sampling with selectable N.
- Compare validation plots.

Run:
```bash
pip install -r requirements.txt
streamlit run app.py
```
