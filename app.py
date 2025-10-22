# app.py — 2D Tile Temperature (v2)
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.tiletemp.physics import tile_temperature_field, eich_1d_profile
from src.tiletemp.utils import fixed_grids, pack_controls, outputs_to_row_2d, lhs, halton, NX_FIXED, NZ_FIXED

st.set_page_config(page_title="Divertor Tile Temperature — 2D (x–z) — v2", layout="wide")
st.title("Divertor Tile Temperature — 2D (x–z) — v2")
st.caption("Fixed strike offset; fixed grids (x: -50→150 mm, Nx=50; z: 0→20 mm, Nz=40). Adds neutral pressure (broadening) and impurity fraction (power removal).")

with st.sidebar:
    st.header("Control parameters")
    P_SOL_MW = st.slider("P_SOL to outer target [MW]", 0.5, 30.0, 5.0, 0.5)
    flux_expansion = st.slider("Flux expansion [-]", 1.0, 15.0, 5.0, 0.5)
    angle_deg = st.slider("Incidence angle [deg]", 0.5, 5.0, 2.0, 0.1)
    neutral_pressure = st.slider("Neutral pressure [arb.]", 0.0, 5.0, 0.0, 0.1)
    impurity_fraction = st.slider("Impurity fraction (power removed) [-]", 0.0, 0.5, 0.0, 0.01)

    st.header("Tile & cooling")
    coolant_T_K = st.slider("Coolant temperature T_c [K]", 273.0, 700.0, 350.0, 1.0)
    k_W_mK = st.slider("Thermal conductivity k [W/m·K]", 30.0, 250.0, 120.0, 1.0)
    thickness_mm = st.slider("Tile thickness [mm]", 2.0, 60.0, 20.0, 0.5)

    st.divider()
    st.header("Auto-generate dataset")
    use_auto = st.toggle("Auto-generate via sampler")
    if use_auto:
        n_points = st.number_input("Number of samples (N)", 1, 2000, 100, 1)
        sampler = st.selectbox("Sampler", ["Latin Hypercube (LHS)", "Halton (low-discrepancy)"])
        st.caption("LHS is robust for moderate dimensions; Halton offers low-discrepancy coverage (Sobol-like).")
        gen_btn = st.button("Generate & Append")

x, z = fixed_grids()

tab1, tab2 = st.tabs(["Explore & Build", "Compare prediction"])

with tab1:
    T = tile_temperature_field(
        x, z,
        P_SOL_MW=P_SOL_MW,
        flux_expansion=flux_expansion,
        angle_deg=angle_deg,
        coolant_T_K=coolant_T_K,
        k_W_mK=k_W_mK,
        thickness_mm=thickness_mm,
        neutral_pressure=neutral_pressure,
        impurity_fraction=impurity_fraction,
    )
    q_MW_m2 = eich_1d_profile(
        x, P_SOL_MW*(1.0-impurity_fraction), flux_expansion, angle_deg,
        neutral_pressure=neutral_pressure
    )

    col1, col2 = st.columns([3,1], gap="large")
    with col1:
        fig, ax = plt.subplots(figsize=(7.5,5.5))
        im = ax.imshow(T.T, origin='lower', extent=[x.min(), x.max(), z.min(), z.max()], aspect='auto')
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

    if "inputs_df" not in st.session_state:
        st.session_state.inputs_df = pd.DataFrame()
    if "outputs_df" not in st.session_state:
        st.session_state.outputs_df = pd.DataFrame()

    params = dict(
        P_SOL_MW=P_SOL_MW, flux_expansion=flux_expansion, angle_deg=angle_deg,
        coolant_T_K=coolant_T_K, k_W_mK=k_W_mK, thickness_mm=thickness_mm,
        neutral_pressure=neutral_pressure, impurity_fraction=impurity_fraction,
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

    if 'rng' not in st.session_state:
        st.session_state.rng = np.random.default_rng(123)

    if use_auto and gen_btn:
        rng = st.session_state.rng
        n = int(n_points)
        dims = 8
        if sampler.startswith("Latin"):
            U = lhs(n, dims, rng)
        else:
            U = halton(n, dims, rng)

        bounds = dict(
            P_SOL_MW=(0.5, 30.0),
            flux_expansion=(1.0, 15.0),
            angle_deg=(0.5, 5.0),
            coolant_T_K=(273.0, 700.0),
            k_W_mK=(30.0, 250.0),
            thickness_mm=(2.0, 60.0),
            neutral_pressure=(0.0, 5.0),
            impurity_fraction=(0.0, 0.5),
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
        st.success(f"Appended {n} samples via {sampler}.")

    with st.expander("Current dataset (inputs)"):
        st.dataframe(st.session_state.inputs_df, use_container_width=True, height=240)
    with st.expander("Current dataset (outputs: flattened T_ij)"):
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

with tab2:
    st.subheader("Compare model prediction vs. ground truth")
    st.caption("Upload your model's predicted T(x,z) as a CSV (single row or flattened vector; length must be Nx×Nz). Optional: upload per-pixel uncertainty (same shape).")

    T_true = tile_temperature_field(
        x, z,
        P_SOL_MW=P_SOL_MW,
        flux_expansion=flux_expansion,
        angle_deg=angle_deg,
        coolant_T_K=coolant_T_K,
        k_W_mK=k_W_mK,
        thickness_mm=thickness_mm,
        neutral_pressure=neutral_pressure,
        impurity_fraction=impurity_fraction,
    )

    nx, nz = len(x), len(z)
    n_expected = nx * nz

    pred_file = st.file_uploader(f"Prediction CSV (flattened length {n_expected})", type=["csv"])
    unc_file = st.file_uploader("Optional Uncertainty CSV (same length)", type=["csv"])

    if pred_file is not None:
        try:
            arr = np.loadtxt(pred_file, delimiter=",")
            if arr.ndim > 1:
                arr = arr.ravel()
            if arr.size != n_expected:
                st.error(f"Prediction has {arr.size} values; expected {n_expected} (Nx×Nz = {nx}×{nz}).")
            else:
                T_pred = arr.reshape((nx, nz), order="C")
                err = T_pred - T_true
                rmse = float(np.sqrt(np.mean(err**2)))
                mae = float(np.mean(np.abs(err)))
                c1, c2, c3 = st.columns(3, gap="small")
                with c1:
                    st.metric("RMSE [K]", f"{rmse:.3f}")
                    st.metric("MAE  [K]", f"{mae:.3f}")
                with c2:
                    fig, ax = plt.subplots(figsize=(4.6,3.6))
                    im = ax.imshow(T_true.T, origin='lower', extent=[x.min(), x.max(), z.min(), z.max()], aspect='auto')
                    plt.colorbar(im, ax=ax).set_label("T_true [K]")
                    ax.set_title("Ground truth")
                    st.pyplot(fig)
                with c3:
                    fig, ax = plt.subplots(figsize=(4.6,3.6))
                    im = ax.imshow(T_pred.T, origin='lower', extent=[x.min(), x.max(), z.min(), z.max()], aspect='auto')
                    plt.colorbar(im, ax=ax).set_label("T_pred [K]")
                    ax.set_title("Prediction")
                    st.pyplot(fig)

                fig, ax = plt.subplots(figsize=(6.6,3.6))
                im = ax.imshow((T_pred - T_true).T, origin='lower', extent=[x.min(), x.max(), z.min(), z.max()], aspect='auto')
                plt.colorbar(im, ax=ax).set_label("Error [K]")
                ax.set_title("Error (Pred − True)")
                st.pyplot(fig)

                if unc_file is not None:
                    uarr = np.loadtxt(unc_file, delimiter=",")
                    if uarr.ndim > 1:
                        uarr = uarr.ravel()
                    if uarr.size == n_expected:
                        U = uarr.reshape((nx, nz), order="C")
                        with st.expander("Uncertainty diagnostics"):
                            zmap = (T_pred - T_true) / np.where(U>0, U, np.nan)
                            coverage = np.nanmean(np.abs(zmap) <= 1.0) * 100.0
                            st.write(f"±1σ coverage: **{coverage:.1f}%** (ideal ≈ 68%)")
                            fig, ax = plt.subplots(figsize=(6,3.6))
                            im = ax.imshow(U.T, origin='lower', extent=[x.min(), x.max(), z.min(), z.max()], aspect='auto')
                            plt.colorbar(im, ax=ax).set_label("Uncertainty σ [K]")
                            ax.set_title("Provided Uncertainty (σ)")
                            st.pyplot(fig)
                    else:
                        st.warning(f"Uncertainty has {uarr.size} values; expected {n_expected}. Ignoring.")
        except Exception as e:
            st.error(f"Could not parse prediction: {e}")
