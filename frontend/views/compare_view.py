"""Vista 3: Comparación interactiva de artefactos, mapas 2D, cortes 1D y exportación."""

from __future__ import annotations

import streamlit as st  # type: ignore

from pinn_thermal.core.evaluation import evaluate_model
from pinn_thermal.visualization.grids import GridSampler
from pinn_thermal.visualization.plots import InteractivePlotter, StaticPlotter
from frontend.state import WorkspaceStore


def render_compare_view(store: WorkspaceStore) -> None:
    st.header("3. Comparación de Modelos, Campos Térmicos y Líneas de Corte")

    runs = store.list_runs()
    if not runs:
        st.info("Aún no existen corridas en el workspace compartido. Ejecute un modelo en la pestaña 2 o desde un cuaderno Jupyter.")
        return

    st.subheader("Tabla Resumen de Corridas (Streamlit + Jupyter + CLI)")
    st.dataframe(
        [
            {
                "run_id": r["run_id"],
                "tipo": r["kind"],
                "caso": r["case"],
                "modelo": r["architecture"],
                "estado": r["status"],
                "Tmax [°C]": r["Tmax_C"],
                "Balance [%]": r["balance_pct"],
                "NRMSE [%]": r["nrmse_pct"],
                "Tiempo [s]": r["elapsed_s"],
                "Dispositivo": r["device"],
            }
            for r in runs
        ],
        use_container_width=True,
    )

    run_ids = [r["run_id"] for r in runs]
    selected = st.multiselect("Seleccionar soluciones para visualizar/comparar", run_ids, default=run_ids[: min(3, len(run_ids))])
    if not selected:
        return

    primary_id = st.selectbox("Solución principal para mapa 2D", selected)
    ref_choices = ["(Ninguna — usar exacta analítica si existe)"] + [rid for rid in run_ids if rid != primary_id]
    ref_id = st.selectbox("Referencia de contraste (p. ej. FEM FEniCSx)", ref_choices)

    primary_art = store.load_run(primary_id)
    ref_art = store.load_run(ref_id) if ref_id != ref_choices[0] else None

    c1, c2, c3 = st.columns(3)
    res_nx = c1.slider("Resolución X/Y de grilla", 80, 300, 150)
    show_flux = c2.checkbox("Mostrar vectores de flujo q", value=False)
    show_err = c3.checkbox("Mostrar mapa de error T - T_ref", value=ref_art is not None or primary_art.problem.reference(primary_art.problem._representative_point(primary_art.problem.regions[0]).reshape(1, 2)) is not None)

    grid = GridSampler.sample_artifact_grid(
        primary_art,
        nx=res_nx,
        ny=res_nx,
        with_flux=show_flux,
        reference_predictor=ref_art,
    )
    fig2d = InteractivePlotter.temperature_figure(
        grid,
        primary_art.problem,
        title=f"{primary_id} ({primary_art.architecture})",
        show_flux_quiver=show_flux,
        show_error=show_err,
    )
    st.plotly_chart(fig2d, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})

    # Comparación cuantitativa cruzada si se eligió referencia externa (p. ej. FEM)
    if ref_art is not None and primary_art.kind == "pinn" and primary_art.model is not None and primary_art.problem.physics_sha256 == ref_art.problem.physics_sha256:
        pts, _, _ = primary_art.problem.evaluation(400)
        ref_vals = ref_art.predict(pts)["T"]
        cross_rep, _ = evaluate_model(primary_art.model, reference=ref_vals, reference_label=ref_id, include_audit=False)
        st.json({"comparacion_contra_referencia": cross_rep})

    # Corte 1D multi-modelo
    st.subheader("Perfil 1D a lo largo de una recta (Start -> End)")
    x0, x1, y0, y1 = primary_art.problem.bounds
    lc1, lc2, lc3, lc4 = st.columns(4)
    sx = lc1.number_input("x inicio [m]", value=float(x0))
    sy = lc2.number_input("y inicio [m]", value=float(0.5 * (y0 + y1)))
    ex = lc3.number_input("x fin [m]", value=float(x1))
    ey = lc4.number_input("y fin [m]", value=float(0.5 * (y0 + y1)))

    predictors = {}
    for rid in selected:
        art = store.load_run(rid)
        if art.problem.bounds == primary_art.problem.bounds:
            predictors[f"{rid} ({art.architecture})"] = art
    if predictors:
        cut = GridSampler.sample_line_cut(predictors, primary_art.problem, (sx, sy), (ex, ey))
        st.plotly_chart(InteractivePlotter.line_cut_figure(cut), use_container_width=True)

    # Curvas de convergencia de modelos PINN seleccionados
    histories = {}
    for rid in selected:
        art = store.load_run(rid)
        if art.result.get("history"):
            histories[f"{rid} ({art.architecture})"] = art.result["history"]
    if histories:
        st.plotly_chart(InteractivePlotter.convergence_figure(histories, metric="total"), use_container_width=True)

    if st.button("Exportar lámina PNG estática de publicación en el directorio de la corrida"):
        out_png = store.runs_dir / primary_id / "publicacion_2d.png"
        StaticPlotter.plot_temperature_field(grid, primary_art.problem, title=f"{primary_art.problem.title} ({primary_art.architecture})", output_path=out_png)
        st.success(f"Exportado: `{out_png}`")
