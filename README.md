# demos-tiletemp-2d — v7

**Frozen parameters**
- thickness_mm = 20.0
- k_W_mK = 120.0
- coolant_T_K = 350.0
- angle_deg = 2.0
- flux_expansion = 5.0

**Free controls**: `P_SOL_MW`, `neutral_fraction`, `impurity_fraction`, `ne_19`

- Grid 30×20; LHS sampling with selectable N; headered CSVs; no index in downloads.
- Compare tab with axis labels and shared Error/Uncertainty scales (95% CI = 1.96×σ).
- Temperature → `plasma`, Error → `seismic`, Uncertainty → `Reds`.

Run:
```bash
pip install -r requirements.txt
streamlit run app.py
```
