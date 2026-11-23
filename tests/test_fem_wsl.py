"""Pruebas de integración real con FEniCSx (DolfinX + Gmsh P2) vía WSL2/nativo/Docker."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pinn_thermal.core.artifact import SolutionArtifact
from pinn_thermal.core.trainer import PINNTrainer, TrainConfig
from pinn_thermal.fem.runner import FEniCSxAdapter
from pinn_thermal.visualization.grids import GridSampler


@pytest.mark.fem
def test_fem_solve_and_pinn_transfer(tmp_path: Path, catalog) -> None:
    adapter = FEniCSxAdapter()
    status = adapter.is_available()
    if not status.get("available"):
        pytest.skip(f"Backend FEniCSx no disponible: {status}")

    # 1. Caso MMS analítico: comparar FEM P2 contra solución exacta y validar balance
    mms = catalog.load("mms_constant")
    fem_dir = tmp_path / "fem_mms"
    fem_art = adapter.solve(mms, output_dir=fem_dir, level=1)
    assert fem_art.kind == "fem"
    assert fem_art.result["balance_pct"] < 1.0
    pts, _, _ = mms.evaluation(per_region=120)
    pred_t = fem_art.predict(pts)["T"]
    exact_t = mms.reference(pts)
    rmse = float(np.sqrt(np.mean((pred_t - exact_t) ** 2)))
    assert rmse < 0.05

    # 2. Caso heterogéneo con tabla 2D + piecewise + tabla 1D en Robin + expresión
    demo = catalog.load("tabla_heterogenea_demo")
    fem_demo_dir = tmp_path / "fem_demo"
    demo_art = adapter.solve(demo, output_dir=fem_demo_dir, level=1)
    assert demo_art.result["balance_pct"] < 2.0
    assert np.isfinite(demo_art.result["Tmax_C"])

    # 3. Transferencia funcional FEM -> PINN (CUSPR)
    pinn_dir = tmp_path / "pinn_from_fem"
    art_pinn = PINNTrainer(
        TrainConfig(
            adam_steps=20,
            lbfgs_steps=0,
            width=16,
            depth=2,
            device="cpu",
            warm_start_from=fem_demo_dir,
            transfer_mode="functional_pretrain",
            transfer_steps=25,
            points={"n_interior": 64, "n_layer": 48, "n_face": 24, "n_boundary": 24},
        )
    ).train(demo, "CUSPR", pinn_dir, overwrite=True)
    assert art_pinn.result["status"] == "completed"
    assert art_pinn.result["transfer_report"]["source_kind"] == "fem"


@pytest.mark.fem
def test_fem_annulus_benchmark(tmp_path: Path, catalog) -> None:
    adapter = FEniCSxAdapter()
    status = adapter.is_available()
    if not status.get("available"):
        pytest.skip(f"Backend FEniCSx no disponible: {status}")

    annulus = catalog.load("annulus")
    art = adapter.solve(annulus, output_dir=tmp_path / "fem_annulus_l2", level=2)
    assert art.kind == "fem"
    assert art.result["balance_pct"] < 1.0
    assert abs(art.result["Tmin_C"] - 20.0) < 1e-3
    assert abs(art.result["Tmax_C"] - 49.585722305155578) < 0.1

    pts, _, _ = annulus.evaluation(per_region=200)
    pred_t = art.predict(pts)["T"]
    exact_t = annulus.reference(pts)
    rmse = float(np.sqrt(np.mean((pred_t - exact_t) ** 2)))
    assert rmse < 0.05

    grid = GridSampler.sample_artifact_grid(art, nx=40, ny=40)
    assert grid.valid_mask is not None
    assert np.any(~grid.valid_mask)
    assert np.all(np.isfinite(grid.T[grid.valid_mask]))
    assert np.all(np.isnan(grid.T[~grid.valid_mask]))
