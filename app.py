# app.py — 2D Tile Temperature (v4)
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from src.tiletemp.physics import tile_temperature_field, eich_1d_profile
from src.tiletemp.utils import (
    fixed_grids, pack_controls, outputs_to_row_2d,
    lhs, NX_FIXED, NZ_FIXED, CONTROL_KEYS,
    parse_inputs_csv_with_header, read_flat_with_header
)

st.set_page_config(page_title="Divertor Tile Temperature — 2D (x–z) — v4", layout="wide")
st.title("Divertor Tile Temperature — 2D (x–z) — v4")
st.caption("Fixed smaller grids; LHS auto-sampler; headered CSVs; filename fields for downloads.")

# ---- Sidebar controls ----
with st.sidebar:
    st.header("Control parameters")
    P_SOL_MW = st.slider("P_SOL to outer target [MW]", 0.5, 30.0, 5.0, 0.5)
    flux_expansion = st.slider("Flux expansion [-]", 1.0, 15.0, 5.0, 0.5)
    angle_deg = st.slider("Incidence angle [deg]", 0.5, 5.0, 2.0, 0.1)
    neutral_fraction = st.slider("Neutral fraction [-]", 0.0, 1.0, 0.0, 0.01)
    impurity_fraction = st.slider("Impurity fraction (power removed) [-]", 0.0, 0.5, 0.0, 0.01)
    ne_19 = st.slider("Upstream density nₑ [10¹⁹ m⁻³]", 1.0, 15.0, 5.0, 0.5)

    st.header("Tile & cooling")
    coolant_T_K = st.slider("Coolant temperature T_c [K]", 273.0, 700.0, 350.0, 1.0)
    k_W_mK = st.slider("Thermal conductivity k [W/m·K]", 30.0, 250.0, 120.0, 1.0)
    thickness_mm = st.slider("Tile thickness [mm]", 2.0, 60.0, 20.0, 0.5)

    st.divider()
    st.header("Auto-generate dataset (LHS only)")
    use_auto = st.toggle("Enable LHS auto-generation")
    n_points = st.number_input("Number of LHS samples", 1, 2000, 20, 1)
    gen_btn = st.button("Generate & Append")

# Fixed smaller grids
x, z = fixed_grids()

# Tabs
tab1, tab2 = st.tabs(["Explore & Build", "Compare prediction"])

with tab1:
    # Ground truth from sidebar controls
    T = tile_temperature_field(
        x, z,
        P_SOL_MW=P_SOL_MW,
        flux_expansion=flux_expansion,
        angle_deg=angle_deg,
        coolant_T_K=coolant_T_K,
        k_W_mK=k_W_mK,
        thickness_mm=thickness_mm,
        neutral_fraction=neutral_fraction,
        impurity_fraction=impurity_fraction,
        ne_19=ne_19,
    )
    q_MW_m2 = eich_1d_profile(
        x, P_SOL_MW*(1.0-impurity_fraction), flux_expansion, angle_deg,
        neutral_fraction=neutral_fraction, ne_19=ne_19
    )

    col1, col2 = st.columns([3,1], gap="large")
    with col1:
        fig, ax = plt.subplots(figsize=(7.5,5.5))
        im = ax.imshow(T.T, origin='lower', extent=[x.min(), x.max(), z.min(), z.max()], aspect='auto', cmap='plasma')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Temperature T(x,z) [K]")
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
        st.caption(f"Fixed grid: Nx={NX_FIXED}, Nz={NZ_FIXED}. x=[{x.min():.0f},{x.max():.0f}] mm; z=[{z.min():.0f},{z.max():.0f}] mm.")

    # Dataset state
    if "inputs_df" not in st.session_state:
        st.session_state.inputs_df = pd.DataFrame()
    if "outputs_df" not in st.session_state:
        st.session_state.outputs_df = pd.DataFrame()

    params = dict(
        P_SOL_MW=P_SOL_MW, flux_expansion=flux_expansion, angle_deg=angle_deg,
        coolant_T_K=coolant_T_K, k_W_mK=k_W_mK, thickness_mm=thickness_mm,
        neutral_fraction=neutral_fraction, impurity_fraction=impurity_fraction, ne_19=ne_19,
    )

    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        if st.button("➕ Add row"):
            from src.tiletemp.utils import pack_controls, outputs_to_row_2d
            in_row = pack_controls(params)
            out_row = outputs_to_row_2d(T, prefix="T")
            st.session_state.inputs_df = pd.concat([st.session_state.inputs_df, pd.DataFrame([in_row])], ignore_index=True)
            st.session_state.outputs_df = pd.concat([st.session_state.outputs_df, pd.DataFrame([out_row])], ignore_index=True)
    with c2:
        if st.button("🗑️ Clear dataset"):
            st.session_state.inputs_df = pd.DataFrame()
            st.session_state.outputs_df = pd.DataFrame()

    # Auto-generate via LHS only
    if 'rng' not in st.session_state:
        st.session_state.rng = np.random.default_rng(123)
    if use_auto and gen_btn:
        rng = st.session_state.rng
        n = int(n_points)
        dims = len(CONTROL_KEYS)
        U = lhs(n, dims, rng)

        bounds = dict(
            P_SOL_MW=(0.5, 30.0),
            flux_expansion=(1.0, 15.0),
            angle_deg=(0.5, 5.0),
            coolant_T_K=(273.0, 700.0),
            k_W_mK=(30.0, 250.0),
            thickness_mm=(2.0, 60.0),
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
            T_i = tile_temperature_field(x, z, **vals)
            add_inputs.append(pack_controls(vals))
            from src.tiletemp.utils import outputs_to_row_2d
            add_outputs.append(outputs_to_row_2d(T_i, prefix="T"))

        st.session_state.inputs_df = pd.concat([st.session_state.inputs_df, pd.DataFrame(add_inputs)], ignore_index=True)
        st.session_state.outputs_df = pd.concat([st.session_state.outputs_df, pd.DataFrame(add_outputs)], ignore_index=True)
        st.success(f"Appended {n} samples via Latin Hypercube.")

    # Download filename fields
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        fname_inputs = st.text_input("Inputs filename", "inputs.csv")
        st.download_button(
            "⬇️ Download inputs.csv",
            data=st.session_state.inputs_df.to_csv(index=True).encode("utf-8"),
            file_name=fname_inputs,
            mime="text/csv",
            disabled=st.session_state.inputs_df.empty,
        )
    with col_dl2:
        fname_outputs = st.text_input("Outputs filename", "outputs.csv")
        st.download_button(
            "⬇️ Download outputs.csv",
            data=st.session_state.outputs_df.to_csv(index=True).encode("utf-8"),
            file_name=fname_outputs,
            mime="text/csv",
            disabled=st.session_state.outputs_df.empty,
        )

with tab2:
    st.subheader("Compare model prediction vs. ground truth")
    st.caption("All CSVs must have a header row. Upload inputs CSV (headers match controls), prediction CSV (flattened), and uncertainty CSV (flattened 1σ).")

    # Inputs CSV (with headers) to set ground truth
    inputs_file = st.file_uploader("Inputs CSV (with headers)", type=["csv"], key="inputs_csv_v4")
    if inputs_file is not None:
        try:
            vals = parse_inputs_csv_with_header(inputs_file)
            # Fill missing keys from sidebar defaults
            for k in CONTROL_KEYS:
                if k not in vals:
                    vals[k] = locals().get(k, None)
            P_SOL_MW_c = float(vals["P_SOL_MW"])
            flux_expansion_c = float(vals["flux_expansion"])
            angle_deg_c = float(vals["angle_deg"])
            coolant_T_K_c = float(vals["coolant_T_K"])
            k_W_mK_c = float(vals["k_W_mK"])
            thickness_mm_c = float(vals["thickness_mm"])
            neutral_fraction_c = float(vals["neutral_fraction"])
            impurity_fraction_c = float(vals["impurity_fraction"])
            ne_19_c = float(vals["ne_19"])
        except Exception as e:
            st.error(f"Could not parse inputs CSV: {e}")
            inputs_file = None

    if inputs_file is None:
        P_SOL_MW_c = P_SOL_MW
        flux_expansion_c = flux_expansion
        angle_deg_c = angle_deg
        coolant_T_K_c = coolant_T_K
        k_W_mK_c = k_W_mK
        thickness_mm_c = thickness_mm
        neutral_fraction_c = neutral_fraction
        impurity_fraction_c = impurity_fraction
        ne_19_c = ne_19

    # Compute ground truth
    T_true = tile_temperature_field(
        x, z,
        P_SOL_MW=P_SOL_MW_c,
        flux_expansion=flux_expansion_c,
        angle_deg=angle_deg_c,
        coolant_T_K=coolant_T_K_c,
        k_W_mK=k_W_mK_c,
        thickness_mm=thickness_mm_c,
        neutral_fraction=neutral_fraction_c,
        impurity_fraction=impurity_fraction_c,
        ne_19=ne_19_c,
    )

    nx, nz = len(x), len(z)
    n_expected = nx * nz

    pred_file = st.file_uploader(f"Prediction CSV (flattened, length {n_expected})", type=["csv"], key="pred_csv_v4")
    unc_file = st.file_uploader("Uncertainty CSV (flattened 1σ, same length)", type=["csv"], key="unc_csv_v4")

    if pred_file is not None and unc_file is not None:
        try:
            pred_arr = read_flat_with_header(pred_file)
            unc_arr = read_flat_with_header(unc_file)

            if pred_arr.size != n_expected or unc_arr.size != n_expected:
                st.error(f"Lengths mismatch. Pred={pred_arr.size}, Unc={unc_arr.size}, expected {n_expected}.")
            else:
                T_pred = pred_arr.reshape((nx, nz), order="C")
                sigma = unc_arr.reshape((nx, nz), order="C")  # 1σ

                # Metrics
                err = T_pred - T_true
                rmse = float(np.sqrt(np.mean(err**2)))
                mae = float(np.mean(np.abs(err)))
                ss_res = float(np.sum(err**2))
                ss_tot = float(np.sum((T_true - np.mean(T_true))**2))
                r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else float('nan')
                var = np.clip(sigma**2, 1e-12, None)
                msll = float(np.mean(0.5*np.log(2*np.pi*var) + 0.5*((T_true - T_pred)**2)/var))

                # Limits
                tmin = float(min(T_true.min(), T_pred.min()))
                tmax = float(max(T_true.max(), T_pred.max()))
                emax = float(np.max(np.abs(err)))
                ci95 = 1.96 * sigma
                u95_max = float(np.percentile(ci95, 95))

                # 2x2 grid
                cA, cB = st.columns(2, gap="large")
                with cA:
                    fig, ax = plt.subplots(figsize=(6,4.2))
                    im = ax.imshow(T_true.T, origin='lower', extent=[x.min(), x.max(), z.min(), z.max()],
                                   aspect='auto', cmap='plasma', vmin=tmin, vmax=tmax)
                    plt.colorbar(im, ax=ax).set_label("T_true [K]")
                    ax.set_title("Ground truth")
                    st.pyplot(fig)
                with cB:
                    fig, ax = plt.subplots(figsize=(6,4.2))
                    im = ax.imshow(T_pred.T, origin='lower', extent=[x.min(), x.max(), z.min(), z.max()],
                                   aspect='auto', cmap='plasma', vmin=tmin, vmax=tmax)
                    plt.colorbar(im, ax=ax).set_label("T_pred [K]")
                    ax.set_title("Prediction")
                    st.pyplot(fig)

                cC, cD = st.columns(2, gap="large")
                with cC:
                    fig, ax = plt.subplots(figsize=(6,4.2))
                    norm = TwoSlopeNorm(vmin=-emax, vcenter=0.0, vmax=emax)
                    im = ax.imshow((T_pred - T_true).T, origin='lower', extent=[x.min(), x.max(), z.min(), z.max()],
                                   aspect='auto', cmap='seismic', norm=norm)
                    plt.colorbar(im, ax=ax).set_label("Error [K] (Pred − True)")
                    ax.set_title("Error (symmetric)")
                    st.pyplot(fig)
                with cD:
                    fig, ax = plt.subplots(figsize=(6,4.2))
                    im = ax.imshow(ci95.T, origin='lower', extent=[x.min(), x.max(), z.min(), z.max()],
                                   aspect='auto', cmap='Reds', vmin=0.0, vmax=u95_max)
                    plt.colorbar(im, ax=ax).set_label("Uncertainty (95% CI) [K]")
                    ax.set_title("Uncertainty (95% CI)")
                    st.pyplot(fig)

                with st.expander("Validation metrics"):
                    st.write(f"**RMSE [K]:** {rmse:.3f}   |   **MAE [K]:** {mae:.3f}   |   **R²:** {r2:.3f}   |   **MSLL:** {msll:.3f}")
        except Exception as e:
            st.error(f"Could not parse uploaded files: {e}")
    else:
        st.info("Please upload both a Prediction CSV and an Uncertainty CSV (with headers) to enable comparison.")
