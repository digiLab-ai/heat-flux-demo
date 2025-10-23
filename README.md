# demos-tiletemp-2d — v6

**What’s new**
- No row index in any CSV downloads (`to_csv(index=False)`).
- LHS auto-generation only, no enable toggle. Just pick **N** and click **Generate & Append**.

**Still in place**
- Fixed grid: **30×10** on x ∈ [-50,150] mm, z ∈ [0,20] mm.
- Inputs/Prediction/Uncertainty CSVs expect headers.
- Compare tab with axis labels and shared Error/Uncertainty scale (Error: symmetric ±V; Uncertainty: [0,V], V from max(|error|) and 95% CI).

**Run**
```bash
pip install -r requirements.txt
streamlit run app.py
```
