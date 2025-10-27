# app.py — 2D Tile Temperature (v7)
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from src.tiletemp.physics import tile_temperature_field, eich_1d_profile
from src.tiletemp.utils import (
    fixed_grids, pack_controls, outputs_to_row_2d,
    lhs, NX_FIXED, NZ_FIXED, CONTROL_KEYS,
    read_flat_with_header
)

st.set_page_config(page_title="Divertor Tile Temperature", layout="wide")
st.title("Divertor Tile Temperature")
st.caption("A simulator for predicting divertor target tile temperature.")

# Frozen constants
THICKNESS_MM = 20.0
K_W_MK = 120.0
COOLANT_T_K = 350.0
ANGLE_DEG = 2.0
FLUX_EXPANSION = 5.0

# Sidebar: only unfrozen
with st.sidebar:
    st.header("Control parameters")
    P_SOL_MW = st.slider("Power to outer SOL [MW]", 0.5, 30.0, 5.0, 0.5)
    neutral_fraction = st.slider("Neutral fraction", 0.0, 1.0, 0.0, 0.01)
    impurity_fraction = st.slider("Impurity fraction", 0.0, 0.5, 0.0, 0.01)
    ne_19 = st.slider("Upstream density nₑ [10¹⁹ m⁻³]", 1.0, 15.0, 5.0, 0.5)

    st.divider()
    st.header("Auto-generate dataset (LHS)")
    n_points = st.number_input("Number of LHS samples", 1, 2000, 20, 1)
    gen_btn = st.button("Generate & Append")

x, z = fixed_grids()
tab1, tab2 = st.tabs(["Explore & Build", "Compare prediction"])

with tab1:
    T = tile_temperature_field(
        x, z,
        P_SOL_MW=P_SOL_MW,
        flux_expansion=FLUX_EXPANSION,
        angle_deg=ANGLE_DEG,
        coolant_T_K=COOLANT_T_K,
        k_W_mK=K_W_MK,
        thickness_mm=THICKNESS_MM,
        neutral_fraction=neutral_fraction,
        impurity_fraction=impurity_fraction,
        ne_19=ne_19,
    )
    q_MW_m2 = eich_1d_profile(
        x, P_SOL_MW*(1.0-impurity_fraction), FLUX_EXPANSION, ANGLE_DEG,
        neutral_fraction=neutral_fraction, ne_19=ne_19
    )

    col1, col2 = st.columns([3,1], gap="large")
    with col1:
        fig, ax = plt.subplots(figsize=(7.5,5.5))
        im = ax.imshow(T.T, origin='lower', extent=[x.min(), x.max(), z.min(), z.max()], aspect='auto', cmap='plasma')
        plt.colorbar(im, ax=ax).set_label("Temperature T(x,z) [K]")
        ax.set_xlabel("x (strike) [mm]")
        ax.set_ylabel("z (depth) [mm]")
        ax.set_title("2D Tile Temperature (ground truth)")
        st.pyplot(fig)

    with col2:
        st.subheader("Surface heat flux")
        fig2, ax2 = plt.subplots(figsize=(4,2.4))
        ax2.plot(x, q_MW_m2, lw=2)
        ax2.set_xlabel("x [mm]")
        ax2.set_ylabel("q(x) [MW/m²]")
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
        st.caption(f"Grid: Nx={NX_FIXED}, Nz={NZ_FIXED}. x=[{x.min():.0f},{x.max():.0f}] mm; z=[{z.min():.0f},{z.max():.0f}] mm.")

    if "inputs_df" not in st.session_state:
        st.session_state.inputs_df = pd.DataFrame()
    if "outputs_df" not in st.session_state:
        st.session_state.outputs_df = pd.DataFrame()

    params = dict(P_SOL_MW=P_SOL_MW, neutral_fraction=neutral_fraction, impurity_fraction=impurity_fraction, ne_19=ne_19)

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        if st.button("➕ Add row"):
            in_row = pack_controls(params)
            out_row = outputs_to_row_2d(T, prefix="T")
            st.session_state.inputs_df = pd.concat([st.session_state.inputs_df, pd.DataFrame([in_row])], ignore_index=True)
            st.session_state.outputs_df = pd.concat([st.session_state.outputs_df, pd.DataFrame([out_row])], ignore_index=True)
    with c2:
        if st.button("🗑️ Clear dataset"):
            st.session_state.inputs_df = pd.DataFrame()
            st.session_state.outputs_df = pd.DataFrame()
    count_placeholder = c3.empty()

    if 'rng' not in st.session_state:
        st.session_state.rng = np.random.default_rng(123)
    if gen_btn:
        rng = st.session_state.rng
        n = int(n_points)
        dims = len(CONTROL_KEYS)
        U = lhs(n, dims, rng)
        bounds = dict(
            P_SOL_MW=(0.5, 30.0),
            neutral_fraction=(0.0, 1.0),
            impurity_fraction=(0.0, 0.5),
            ne_19=(1.0, 15.0),
        )
        keys = list(bounds.keys())
        lo = np.array([bounds[k][0] for k in keys])
        hi = np.array([bounds[k][1] for k in keys])
        samples = lo + U * (hi - lo)

        add_inputs, add_outputs = [], []
        for row in samples:
            vals = dict(zip(keys, row))
            T_i = tile_temperature_field(
                x, z,
                P_SOL_MW=vals["P_SOL_MW"],
                flux_expansion=FLUX_EXPANSION,
                angle_deg=ANGLE_DEG,
                coolant_T_K=COOLANT_T_K,
                k_W_mK=K_W_MK,
                thickness_mm=THICKNESS_MM,
                neutral_fraction=vals["neutral_fraction"],
                impurity_fraction=vals["impurity_fraction"],
                ne_19=vals["ne_19"],
            )
            add_inputs.append(pack_controls(vals))
            add_outputs.append(outputs_to_row_2d(T_i, prefix="T"))

        st.session_state.inputs_df = pd.concat([st.session_state.inputs_df, pd.DataFrame(add_inputs)], ignore_index=True)
        st.session_state.outputs_df = pd.concat([st.session_state.outputs_df, pd.DataFrame(add_outputs)], ignore_index=True)
        st.success(f"Appended {n} samples via Latin Hypercube.")

    count_placeholder.metric("Samples in dataset:", int(len(st.session_state.inputs_df)))

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        fname_inputs = st.text_input("Inputs filename", "inputs_train.csv")
        st.download_button(
            "⬇️ Download inputs",
            data=st.session_state.inputs_df.to_csv(index=False).encode("utf-8"),
            file_name=fname_inputs,
            mime="text/csv",
            disabled=st.session_state.inputs_df.empty,
        )
    with col_dl2:
        fname_outputs = st.text_input("Outputs filename", "outputs_train.csv")
        st.download_button(
            "⬇️ Download outputs",
            data=st.session_state.outputs_df.to_csv(index=False).encode("utf-8"),
            file_name=fname_outputs,
            mime="text/csv",
            disabled=st.session_state.outputs_df.empty,
        )

    with st.expander("Show inputs dataset", expanded=False):
        if st.session_state.inputs_df.empty:
            st.info("No input rows have been added yet.")
        else:
            st.dataframe(st.session_state.inputs_df)

    with st.expander("Show outputs dataset", expanded=False):
        if st.session_state.outputs_df.empty:
            st.info("No output rows have been added yet.")
        else:
            st.dataframe(st.session_state.outputs_df)

with tab2:
    st.subheader("Compare model prediction vs. ground truth")
    st.caption("Please upload the validation outputs, emulator prediction, and emulator uncertainty. Expected flattened length = 600 (=30×20).")

    nx, nz = len(x), len(z)
    n_expected = nx * nz

    truth_file = st.file_uploader(f"Validation outputs CSV (flattened, length {n_expected})", type=["csv"], key="truth_csv")
    pred_file = st.file_uploader(f"Prediction CSV (flattened, length {n_expected})", type=["csv"], key="pred_csv")
    unc_file = st.file_uploader("Uncertainty CSV (flattened 1σ, same length)", type=["csv"], key="unc_csv")

    T_true = None
    if truth_file is not None:
        try:
            truth_arr = read_flat_with_header(truth_file)
            if truth_arr.size != n_expected:
                st.error(f"Validation outputs length {truth_arr.size} does not match expected {n_expected}.")
            else:
                T_true = truth_arr.reshape((nx, nz), order="C")
        except Exception as e:
            st.error(f"Could not parse validation outputs CSV: {e}")

    if T_true is not None and pred_file is not None and unc_file is not None:
        try:
            pred_arr = read_flat_with_header(pred_file)
            unc_arr = read_flat_with_header(unc_file)

            if pred_arr.size != n_expected or unc_arr.size != n_expected:
                st.error(f"Lengths mismatch. Pred={pred_arr.size}, Unc={unc_arr.size}, expected {n_expected}.")
            else:
                T_pred = pred_arr.reshape((nx, nz), order="C")
                sigma = unc_arr.reshape((nx, nz), order="C")

                err = T_pred - T_true
                rmse = float(np.sqrt(np.mean(err**2)))
                mae = float(np.mean(np.abs(err)))
                ss_res = float(np.sum(err**2))
                ss_tot = float(np.sum((T_true - np.mean(T_true))**2))
                r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else float('nan')
                var = np.clip(sigma**2, 1e-12, None)
                log_loss = 0.5*np.log(2*np.pi*var) + 0.5*((T_true - T_pred)**2)/var
                nlpd = float(np.mean(log_loss))

                baseline_mean = float(np.mean(T_true))
                baseline_var = float(np.var(T_true))
                baseline_var = max(baseline_var, 1e-12)
                baseline_loss = 0.5*np.log(2*np.pi*baseline_var) + 0.5*((T_true - baseline_mean)**2)/baseline_var
                msll = float(np.mean(log_loss - baseline_loss))

                tmin = float(min(T_true.min(), T_pred.min()))
                tmax = float(max(T_true.max(), T_pred.max()))
                emax = float(np.max(np.abs(err)))
                ci95 = 1.96 * sigma
                vmax_shared = max(emax, float(np.max(ci95)))
                cA, cB = st.columns(2, gap="large")
                with cA:
                    fig, ax = plt.subplots(figsize=(6,4.2))
                    im = ax.imshow(T_true.T, origin='lower', extent=[x.min(), x.max(), z.min(), z.max()], aspect='auto', cmap='plasma', vmin=tmin, vmax=tmax)
                    plt.colorbar(im, ax=ax).set_label("T_true [K]")
                    ax.set_xlabel("x (strike) [mm]")
                    ax.set_ylabel("z (depth) [mm]")
                    ax.set_title("Ground truth")
                    st.pyplot(fig)
                with cB:
                    fig, ax = plt.subplots(figsize=(6,4.2))
                    im = ax.imshow(T_pred.T, origin='lower', extent=[x.min(), x.max(), z.min(), z.max()], aspect='auto', cmap='plasma', vmin=tmin, vmax=tmax)
                    plt.colorbar(im, ax=ax).set_label("T_pred [K]")
                    ax.set_xlabel("x (strike) [mm]")
                    ax.set_ylabel("z (depth) [mm]")
                    ax.set_title("Prediction")
                    st.pyplot(fig)

                cC, cD = st.columns(2, gap="large")
                with cC:
                    fig, ax = plt.subplots(figsize=(6,4.2))
                    norm = TwoSlopeNorm(vmin=-vmax_shared, vcenter=0.0, vmax=vmax_shared)
                    im = ax.imshow(err.T, origin='lower', extent=[x.min(), x.max(), z.min(), z.max()], aspect='auto', cmap='seismic', norm=norm)
                    plt.colorbar(im, ax=ax).set_label("Error [K] (Pred − True)")
                    ax.set_xlabel("x (strike) [mm]")
                    ax.set_ylabel("z (depth) [mm]")
                    ax.set_title("Error")
                    st.pyplot(fig)
                with cD:
                    fig, ax = plt.subplots(figsize=(6,4.2))
                    im = ax.imshow(ci95.T, origin='lower', extent=[x.min(), x.max(), z.min(), z.max()], aspect='auto', cmap='Reds', vmin=0.0, vmax=vmax_shared)
                    plt.colorbar(im, ax=ax).set_label("Uncertainty (95% CI) [K]")
                    ax.set_xlabel("x (strike) [mm]")
                    ax.set_ylabel("z (depth) [mm]")
                    ax.set_title("Uncertainty")
                    st.pyplot(fig)

                with st.expander("Validation metrics"):
                    st.write(
                        f"**RMSE [K]:** {rmse:.3f}   |   **MAE [K]:** {mae:.3f}   |   **R²:** {r2:.3f}   |   **NLPD:** {nlpd:.3f}   |   **MSLL:** {msll:.3f}"
                    )
                    st.markdown(
                        "- **RMSE / MAE:** Average magnitude of errors in kelvin; lower values mean predictions hug the truth more closely.\n"
                        "- **R²:** Fraction of variance explained; numbers near 1.0 indicate strong agreement, while 0 or negative implies the model misses most structure.\n"
                        "- **NLPD:** Average negative log predictive density; smaller is better because it rewards confident, accurate predictions and penalizes over- or under-confident ones.\n"
                        "- **MSLL:** Mean standardized log loss relative to a constant baseline; negative scores beat the naive baseline, zero matches it, and positive scores underperform."
                    )
        except Exception as e:
            st.error(f"Could not parse prediction or uncertainty CSV: {e}")
    else:
        st.info("Please upload validation outputs, prediction, and uncertainty CSV files (with headers) to enable comparison.")

st.markdown("---")
st.markdown(f"""
### Parameters & assumptions
**Free (control) parameters**:
- Power to outer SOL [MW] - `P_SOL_MW` — power entering the outer scrape off layer (SOL) and travelling to outer target [MW].
- Neutral fraction - `neutral_fraction` — the fraction of neutral species in the divertor plasma. High fractions will increase spreading of heat flux profile.
- Impurity fraction - `impurity_fraction` — the fraction of impurity specicies in the divertor plasma. High fractions remove significant power before reaching the target.
- Upstream density (10¹⁹ m⁻³) - `ne_19` —  the electron density in the scrape off layer at the midplane. High SOL densities will broaden the heat flux profile at the target.
            
**Assumed (fixed) parameters**:
- Tile thickness **t = {20.0} mm**
- Thermal conductivity **k = {120.0} W/m·K**
- Coolant temperature **T_c = {350.0} K**
- Incidence angle **θ = {2.0}°**
- Flux expansion **f_exp = {5.0}**

**Model**: 
- Eich 1D footprint (toroidally symmetric), 
- Steady 1D conduction in depth to a fixed coolant back-face.
""")
