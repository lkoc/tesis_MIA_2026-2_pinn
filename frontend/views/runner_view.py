"""Vista 2: Ejecución de modelos PINN, transferencia, encadenamiento, FEniCSx y ampacidad."""

from __future__ import annotations

import streamlit as st  # type: ignore

import pinn_thermal.models  # noqa: F401
from pinn_thermal.core.ampacity import AmpacitySolver
from pinn_thermal.core.base_model import ModelRegistry
from pinn_thermal.core.device import DeviceManager
from pinn_thermal.core.trainer import PINNTrainer, TrainConfig
from pinn_thermal.core.transfer import CompatibilityChecker
from pinn_thermal.fem.runner import FEniCSxAdapter
from frontend.state import WorkspaceStore


def render_runner_view(store: WorkspaceStore) -> None:
    st.header("2. Entrenamiento PINN, Transferencia entre Redes y Referencia FEM")

    problems = store.list_all_problems()
    if not problems:
        st.warning("No se encontraron casos disponibles.")
        return

    labels = [f"[{p['category']}] {p['id']} — {p['title']}" for p in problems]
    p_idx = st.selectbox("Problema físico", range(len(problems)), format_func=lambda i: labels[i])
    problem = store.load_problem(problems[p_idx]["path"])

    solver_kind = st.radio("Motor a ejecutar", ["Red Neuronal PINN", "Referencia FEM (FEniCSx)", "Cálculo de Ampacidad Máxima"], horizontal=True)

    if solver_kind == "Referencia FEM (FEniCSx)":
        level = st.slider("Nivel de refinamiento de malla Gmsh + DolfinX P2", min_value=1, max_value=4, value=2)
        if st.button("Ejecutar FEniCSx", type="primary"):
            run_dir = store.make_run_dir(problem.id, f"FEM_L{level}")
            with st.spinner("Ejecutando malla Gmsh y solucionador DolfinX P2 en WSL/nativo..."):
                art = FEniCSxAdapter().solve(problem, output_dir=run_dir, level=level)
            st.success(f"FEM completado en `{run_dir.name}` — Tmax = {art.result['Tmax_C']:.3f} °C, Balance = {art.result['balance_pct']:.4f}%")
        return

    if solver_kind == "Cálculo de Ampacidad Máxima":
        runs = [r for r in store.list_runs() if r["case"] == problem.id or True]
        st.info("El cálculo de ampacidad busca mediante bisección la potencia máxima permisible para respetar Tmax objetivo usando PINN o FEM.")
        t_target = st.number_input("Temperatura máxima admisible del conductor [°C]", value=90.0)
        method_amp = st.selectbox("Evaluador térmico", ["PINN rápido (M1_EXACT / CUSPR)", "FEniCSx"])
        if st.button("Calcular Ampacidad Admisible", type="primary"):
            solver = AmpacitySolver(target_temp_C=t_target)
            with st.spinner("Resolviendo potencia máxima mediante bisección térmica..."):
                if method_amp == "FEniCSx":
                    res = solver.solve_with_fem(problem, work_dir=store.runs_dir / "_tmp_ampacity", level=1)
                else:
                    cfg = TrainConfig(adam_steps=300, lbfgs_steps=20, width=24, depth=3)
                    res = solver.solve_with_pinn(problem, architecture="CUSPR", config=cfg, work_dir=store.runs_dir / "_tmp_ampacity")
            st.success(
                f"Corriente I_mid = {res.I_mid_A:.2f} A | Pérdida lineal = {res.power_W_m:.2f} W/m | "
                f"Tmax = {res.Tmax_at_mid_C:.2f} °C | Convergencia = {res.converged}"
            )
        return

    # Entrenamiento / Continuación / Transferencia / Encadenamiento PINN
    models = ModelRegistry.list_models()
    col1, col2, col3 = st.columns(3)
    with col1:
        arch = st.selectbox("Arquitectura PINN", models, index=models.index("M1_EXACT") if "M1_EXACT" in models else 0)
        meta = ModelRegistry.get_metadata(arch)
        st.caption(f"Autores: {meta.authors} ({meta.year}) — {meta.doi or 'tesis'}")
    with col2:
        hw = DeviceManager.describe()
        dev_opts = ["auto", *[name for name, info in hw["backends"].items() if info.get("available")]]
        device = st.selectbox("Dispositivo hardware", dev_opts)
        dtype = st.selectbox("Precisión numérica", ["float32", "float64"])
    with col3:
        activation = st.selectbox("Activación", ["tanh", "silu", "gelu", "sin"])
        initializer = st.selectbox("Inicializador discreto previo", [None, "rbf_readout", "rbf_pretrain", "fem_readout", "fem_pretrain"])

    with st.expander("Hiperparámetros de entrenamiento y muestreo"):
        h1, h2, h3, h4 = st.columns(4)
        adam_steps = h1.number_input("Pasos Adam", min_value=0, max_value=50000, value=600)
        lbfgs_steps = h2.number_input("Pasos L-BFGS", min_value=0, max_value=2000, value=40)
        width = h3.number_input("Ancho de capa (`width`)", min_value=8, max_value=256, value=32)
        depth = h4.number_input("Capas ocultas (`depth`)", min_value=1, max_value=10, value=3)
        focused = st.checkbox("Muestreo focalizado cerca de cables", value=problem.family == "cable")
        attention = st.selectbox("Atención adaptativa (requiere L-BFGS=0)", [None, "rba", "sa"])

    # Continuación o transferencia desde otro artefacto compatible o de distinto JSON
    existing_runs = store.list_runs()
    run_choices = ["(Ninguno — entrenar desde inicialización fresca)"] + [r["run_id"] for r in existing_runs]
    cont_pick = st.selectbox("Reutilizar / Continuar desde solución anterior (Jupyter, CLI o Streamlit)", run_choices)
    cont_kind = "fresh"
    transfer_mode = "auto"
    chain_mode = "residual"
    if cont_pick != run_choices[0]:
        c_col1, c_col2 = st.columns(2)
        cont_kind = c_col1.selectbox(
            "Estrategia de reutilización",
            ["warm_start (pesos o transferencia funcional)", "chain (encadenar salida previa como entrada/base)", "exact_resume (reanudar optimizador exacto)"],
        )
        if cont_kind.startswith("warm_start"):
            transfer_mode = c_col2.selectbox("Modo de transferencia", ["auto", "weights", "functional_readout", "functional_readout_gls", "functional_pretrain"])
        elif cont_kind.startswith("chain"):
            chain_mode = c_col2.selectbox("Modo de encadenamiento", ["residual", "feature"])

        if st.button("Verificar compatibilidad antes de entrenar"):
            src_art = store.load_run(cont_pick)
            tgt_tmp = ModelRegistry.build(problem, arch, width=int(width), depth=int(depth), activation=activation)
            rep = CompatibilityChecker.check(src_art, tgt_tmp)
            st.json(rep.to_dict())

    if st.button("🚀 Iniciar Entrenamiento PINN", type="primary"):
        run_dir = store.make_run_dir(problem.id, arch)
        cfg = TrainConfig(
            adam_steps=int(adam_steps),
            lbfgs_steps=0 if attention else int(lbfgs_steps),
            width=int(width),
            depth=int(depth),
            activation=activation,
            device=device,
            dtype=dtype,
            initializer=initializer,
            focused=focused,
            resample_every=200 if focused else 0,
            attention=attention,
            resume_from=(store.runs_dir / cont_pick) if cont_kind.startswith("exact_resume") else None,
            warm_start_from=(store.runs_dir / cont_pick) if cont_kind.startswith("warm_start") else None,
            transfer_mode=transfer_mode,
            chain_from=(store.runs_dir / cont_pick) if cont_kind.startswith("chain") else None,
            chain_mode=chain_mode,
        )
        bar = st.progress(0.0, text="Preparando modelo y nube física...")
        total_steps = max(cfg.adam_steps + cfg.lbfgs_steps, 1)

        def on_log(row, _model):
            frac = min((row["step"] + 1) / total_steps, 1.0)
            bar.progress(frac, text=f"{row['phase'].upper()} paso {row['step']} — pérdida={row['total']:.3e}")

        art = PINNTrainer(cfg).train(problem, arch, output_dir=run_dir, on_log=on_log, overwrite=True)
        bar.progress(1.0, text="Completado")
        ev = art.result.get("evaluation", {})
        bal = (ev.get("balance") or {}).get("global_derived_pct")
        bal_txt = f"{bal:.3f}%" if bal is not None else "N/A"
        st.success(f"Guardado en `{run_dir.name}` | Pérdida óptima={art.result.get('best_loss', float('nan')):.3e} | Tmax={ev.get('Tmax_pred_C', float('nan')):.3f} °C | Balance={bal_txt}")
