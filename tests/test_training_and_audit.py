"""Pruebas de entrenamiento extremo a extremo, auditoría física, RBF-FD y ampacidad."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from pinn_thermal.core.ampacity import AmpacitySolver
from pinn_thermal.core.artifact import SolutionArtifact
from pinn_thermal.core.problem import ThermalProblem2D
from pinn_thermal.core.trainer import PINNTrainer, TrainConfig


@pytest.mark.parametrize("arch", ["M1", "M2", "CUSPR", "FBPINN", "M1_EXACT"])
def test_short_training_and_audit(tmp_path: Path, mms_problem: ThermalProblem2D, arch: str) -> None:
    out_dir = tmp_path / f"run_{arch}"
    cfg = TrainConfig(
        adam_steps=40,
        lbfgs_steps=5,
        log_every=20,
        width=16,
        depth=2,
        device="cpu",
        dtype="float32",
        points={"n_interior": 64, "n_layer": 48, "n_face": 24, "n_boundary": 24},
    )
    art = PINNTrainer(cfg).train(mms_problem, arch, out_dir, overwrite=True)
    assert art.result["status"] == "completed"
    assert (out_dir / "model.pt").is_file()
    assert (out_dir / "result.json").is_file()
    assert (out_dir / "manifest.json").is_file()
    ev = art.result["evaluation"]
    assert ev["rmse_K"] >= 0.0
    assert "balance" in ev

    loaded = SolutionArtifact.load(out_dir)
    pts, _, _ = mms_problem.evaluation(per_region=20)
    diff = abs(art.predict(pts)["T"] - loaded.predict(pts)["T"]).max()
    assert diff < 1e-5


def test_rbffd_initializer_and_attention_rba(tmp_path: Path, heterogeneous_problem: ThermalProblem2D) -> None:
    out_dir = tmp_path / "run_rbf_rba"
    cfg = TrainConfig(
        adam_steps=30,
        lbfgs_steps=0,  # RBA requiere lbfgs_steps=0
        width=16,
        depth=2,
        device="cpu",
        attention="rba",
        initializer="rbf_readout",
        initializer_settings={"n_per_region": 80, "stencil_size": 15},
        points={"n_interior": 64, "n_layer": 48, "n_face": 24, "n_boundary": 24},
    )
    art = PINNTrainer(cfg).train(heterogeneous_problem, "CUSPR", out_dir, overwrite=True)
    assert art.result["status"] == "completed"
    assert art.result["initializer_report"]["initializer"] == "rbf"
    assert "attention_weights" in art.result


def test_ampacity_coupled_electrothermal_math() -> None:
    """Verifica el punto fijo R(T) y la bisección acotada con un modelo térmico lineal exacto."""
    solver = AmpacitySolver(R20_ohm_m=3.0e-5, T_limit_C=90.0, T_ref_C=20.0)
    res = solver.find_ampacity(initial_upper_A=500.0, field_solver=lambda q: (20.0 + 1.25 * q, 18.0 + 1.15 * q))
    assert res.converged is True
    assert abs(res.Tmax_at_mid_C - 90.0) <= 0.05
    assert 1000.0 < res.I_mid_A < 1500.0


def test_ampacity_solver_pinn(tmp_path: Path, cable_problem: ThermalProblem2D) -> None:
    cfg = TrainConfig(
        adam_steps=0,
        lbfgs_steps=5,
        lbfgs_iterations=5,
        width=16,
        depth=2,
        device="cpu",
        initializer="rbf_readout",
        initializer_settings={"n_per_region": 80, "stencil_size": 15},
        points={"n_interior": 48, "n_layer": 32, "n_face": 16, "n_boundary": 16},
    )
    res = AmpacitySolver(
        R20_ohm_m=1.0e-4,
        target_temp_C=55.0,
        relaxation=0.8,
        max_iters=2,
        coupling_tol_K=1.0,
        source_tol_rel=0.05,
        max_coupling_iters=25,
    ).solve_with_pinn(
        cable_problem, architecture="CUSPR", config=cfg, work_dir=tmp_path / "amp", initial_upper_A=700.0
    )
    assert res.I_mid_A > 0.0
    assert res.power_W_m > 0.0
    assert math.isfinite(res.Tmax_at_mid_C)
    assert len(res.trace) >= 1
