# app.py — 2D Tile Temperature under Eich Heat Flux
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.tiletemp.physics import tile_temperature_field, eich_1d_profile
from src.tiletemp.utils import default_grids, pack_controls, outputs_to_row_2d

st.set_page_config(page_title="Divertor Tile Temperature — 2D (x–z)", layout="wide")
st.title("Divertor Tile Temperature — 2D (x–z) under Eich Heat Flux")
st.caption("Outputs: Temperature field T(x,z) from steady conduction with Eich heat-flux boundary at z=0. Controls are limited to operator-relevant knobs.")

with st.sidebar:
    st.header("Control parameters")
    P_SOL_MW = st.slider("P_SOL to outer target [MW]", 0.5, 30.0, 5.0, 0.5)
    flux_expansion = st.slider("Flux expansion [-]", 1.0, 15.0, 5.0, 0.5)
    angle_deg = st.slider("Incidence angle [deg]", 0.5, 5.0, 2.0, 0.1)
    x0_mm = st.slider("Strike offset x₀ [mm]", -50.0, 50.0, 0.0, 0.5)

    st.header("Tile & cooling")
    coolant_T_K = st.slider("Coolant temperature T_c [K]", 273.0, 700.0, 350.0, 1.0)
    k_W_mK = st.slider("Thermal conductivity k [W/m·K]", 30.0, 250.0, 120.0, 1.0)
    thickness_mm = st.slider("Tile thickness [mm]", 2.0, 60.0, 20.0, 0.5)

    st.divider()
    st.header("Grids")
    x_min_mm, x_max_mm = st.slider("x-range [mm] (strike)", -200.0, 200.0, (-100.0, 100.0), 1.0)
    nx = st.slider("Nx (x resolution)", 50, 800, 300, 10)
    z_min_mm, z_max_mm = st.slider("z-range [mm] (depth)", 0.0, 100.0, (0.0, 20.0), 0.5)
    nz = st.slider("Nz (z resolution)", 40, 800, 120, 10)

x = np.linspace(x_min_mm, x_max_mm, nx)
z = np.linspace(z_min_mm, z_max_mm, nz)

# Compute temperature field
T = tile_temperature_field(
    x, z,
    P_SOL_MW=P_SOL_MW,
    flux_expansion=flux_expansion,
    angle_deg=angle_deg,
    x0_mm=x0_mm,
    coolant_T_K=coolant_T_K,
    k_W_mK=k_W_mK,
    thickness_mm=thickness_mm,
)

# Also show the boundary heat flux q(x)
q_MW_m2 = eich_1d_profile(x, P_SOL_MW, flux_expansion, angle_deg, x0_mm)

col1, col2 = st.columns([3,1], gap="large")
with col1:
    fig, ax = plt.subplots(figsize=(7.5,5.5))
    im = ax.imshow(T.T, origin='lower', extent=[x.min(), x.max(), z.min(), z.max()], aspect='auto')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Temperature T(x,z) [K]")
    ax.set_xlabel("x (strike) [mm]")
    ax.set_ylabel("z (depth) [mm]")
    ax.set_title("2D Tile Temperature")
    st.pyplot(fig)

with col2:
    st.subheader("Surface heat flux")
    fig2, ax2 = plt.subplots(figsize=(4,2.4))
    ax2.plot(x, q_MW_m2, lw=2)
    ax2.set_xlabel("x [mm]")
    ax2.set_ylabel("q(x) [MW/m²]")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

# ---------------- Dataset builder ----------------
if "inputs_df" not in st.session_state:
    st.session_state.inputs_df = pd.DataFrame()
if "outputs_df" not in st.session_state:
    st.session_state.outputs_df = pd.DataFrame()

params = dict(
    P_SOL_MW=P_SOL_MW, flux_expansion=flux_expansion, angle_deg=angle_deg, x0_mm=x0_mm,
    coolant_T_K=coolant_T_K, k_W_mK=k_W_mK, thickness_mm=thickness_mm,
    x_min_mm=x_min_mm, x_max_mm=x_max_mm, nx=nx, z_min_mm=z_min_mm, z_max_mm=z_max_mm, nz=nz,
)

c1, c2, c3 = st.columns([1,1,2])
with c1:
    if st.button("➕ Add row to dataset"):
        from src.tiletemp.utils import pack_controls, outputs_to_row_2d
        in_row = pack_controls(params)
        out_row = outputs_to_row_2d(T, prefix="T")
        st.session_state.inputs_df = pd.concat([st.session_state.inputs_df, pd.DataFrame([in_row])], ignore_index=True)
        st.session_state.outputs_df = pd.concat([st.session_state.outputs_df, pd.DataFrame([out_row])], ignore_index=True)
with c2:
    if st.button("🗑️ Clear dataset"):
        st.session_state.inputs_df = pd.DataFrame()
        st.session_state.outputs_df = pd.DataFrame()

with st.expander("Current dataset (inputs)"):
    st.dataframe(st.session_state.inputs_df, use_container_width=True, height=240)
with st.expander("Current dataset (outputs: flattened T_ij columns)"):
    st.dataframe(st.session_state.outputs_df, use_container_width=True, height=240)

col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    st.download_button(
        "⬇️ Download inputs.csv",
        data=st.session_state.inputs_df.to_csv(index=False).encode("utf-8"),
        file_name="inputs.csv",
        mime="text/csv",
        disabled=st.session_state.inputs_df.empty,
    )
with col_dl2:
    st.download_button(
        "⬇️ Download outputs.csv",
        data=st.session_state.outputs_df.to_csv(index=False).encode("utf-8"),
        file_name="outputs.csv",
        mime="text/csv",
        disabled=st.session_state.outputs_df.empty,
    )

st.divider()
st.markdown("""
### Model summary
- **Flux at surface**: Eich profile \(q(x)\) with configuration-dependent spreading; total toroidal integral ≈ **P_SOL**.
- **Through-tile conduction**: Steady 1D-in-depth conduction to a fixed coolant temperature at the back face.
- **Output field**: Temperature \(T(x,z)\) [K]. Perfect for **2D surrogate** training (e.g., SVD/PCA on images + regressor).
- **Controls only**: P_SOL, flux expansion, incidence angle, strike offset, coolant temperature, conductivity, thickness.
""")
