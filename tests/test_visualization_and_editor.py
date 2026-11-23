"""Pruebas de visualización desacoplada, editor de casos JSON y WorkspaceStore compartido."""

from __future__ import annotations

from pathlib import Path

from pinn_thermal.core.trainer import PINNTrainer, TrainConfig
from pinn_thermal.visualization import CaseEditorHelper, GridSampler, InteractivePlotter, StaticPlotter
from frontend.state import WorkspaceStore


def test_case_editor_helper_and_shared_workspace(tmp_path: Path, cases_root: Path) -> None:
    store = WorkspaceStore(workspace_root=tmp_path / "ws", cases_root=cases_root)
    ed = CaseEditorHelper()
    ed.set_metadata("mi_cable_test", "Cable editado de prueba", "cable")
    ed.upsert_region(
        "backfill",
        kind="soil",
        priority=1,
        shape={"type": "rectangle", "bounds": [-0.3, 0.3, -1.3, -0.7]},
        conductivity=ed.field_patch(1.0, (0.0, -1.0), 0.4, 0.4, 2.2, smoothing=0.05),
    )
    ed.add_annotation(0.0, -1.0, "Centro de conductor")
    assert ed.validate() == []

    saved_case = store.save_user_case(ed)
    assert saved_case.is_file()
    reloaded = store.load_problem("mi_cable_test")
    assert reloaded.id == "mi_cable_test"

    # Entrenar corrida guardada en el WorkspaceStore compartible con Streamlit y Jupyter
    run_dir = store.make_run_dir(reloaded.id, "CUSPR", tag="ui_sync")
    art = PINNTrainer(
        TrainConfig(adam_steps=20, lbfgs_steps=0, width=16, depth=2, device="cpu", points={"n_interior": 40, "n_layer": 24, "n_face": 16, "n_boundary": 16})
    ).train(reloaded, "CUSPR", run_dir, overwrite=True)

    runs = store.list_runs()
    assert len(runs) == 1 and runs[0]["case"] == "mi_cable_test"
    loaded_run = store.load_run(runs[0]["run_id"])

    # Visualización desacoplada (Plotly + Matplotlib)
    grid = GridSampler.sample_artifact_grid(loaded_run, nx=32, ny=32, with_flux=True)
    cut = GridSampler.sample_line_cut({"CUSPR": loaded_run}, reloaded, (0.0, -1.8), (0.0, -0.2), n_points=50)
    assert len(cut["series"]["CUSPR"]) == 50

    fig_geom = InteractivePlotter.geometry_figure(reloaded, background="conductivity", nx=30, ny=30)
    fig_temp = InteractivePlotter.temperature_figure(grid, reloaded, show_flux_quiver=True)
    fig_cut = InteractivePlotter.line_cut_figure(cut)
    fig_conv = InteractivePlotter.convergence_figure({"CUSPR": art.result["history"]})
    assert len(fig_geom.data) >= 1 and len(fig_geom.layout.annotations) == 2
    assert len(fig_temp.data) >= 2
    assert len(fig_cut.data) == 1 and len(fig_conv.data) == 1

    png_out = tmp_path / "static_map.png"
    StaticPlotter.plot_temperature_field(grid, reloaded, output_path=png_out, dpi=90)
    assert png_out.is_file() and png_out.stat().st_size > 1000
