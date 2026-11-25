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


def test_interface_flux_continuity_enforced_in_objective(cable_problem: ThermalProblem2D) -> None:
    """Verifica que interface_flux está activo por defecto (incluso en M1_EXACT) y penaliza saltos [q·n]."""
    import torch
    import pinn_thermal.models  # noqa: F401
    from pinn_thermal.core.base_model import ModelRegistry
    from pinn_thermal.core.objective import DEFAULT_WEIGHTS, PhysicsObjective

    assert DEFAULT_WEIGHTS["interface_flux"] > 0.0
    torch.manual_seed(42)
    points = {"n_interior": 64, "n_layer": 32, "n_face": 24, "n_boundary": 24}

    for arch in ("M1", "M1_EXACT"):
        model = ModelRegistry.build(cable_problem, arch, width=16, depth=2).to(dtype=torch.float32)
        obj_flux = PhysicsObjective(model, **points, exact_energy=False)
        obj_no_flux = PhysicsObjective(model, **points, weights={"interface_flux": 0.0}, exact_energy=False)
        assert obj_flux.weights["interface_flux"] == pytest.approx(DEFAULT_WEIGHTS["interface_flux"])
        assert obj_no_flux.weights["interface_flux"] == 0.0

        total_with, parts_with, _ = obj_flux()
        total_without, parts_without, _ = obj_no_flux()
        flux_term = float(parts_with["interface_flux"].item())
        assert math.isfinite(flux_term) and flux_term > 0.0
        assert float(parts_without["interface_flux"].item()) == pytest.approx(flux_term, rel=1e-5)
        diff = float((total_with - total_without).item())
        assert diff == pytest.approx(DEFAULT_WEIGHTS["interface_flux"] * flux_term, rel=1e-4)

    model_exact = ModelRegistry.build(cable_problem, "M1_EXACT", width=16, depth=2).to(dtype=torch.float32)
    obj_exact_energy = PhysicsObjective(
        model_exact, **points, weights={"energy": 100.0}, exact_energy=True
    )
    assert obj_exact_energy.weights["interface_flux"] == pytest.approx(DEFAULT_WEIGHTS["interface_flux"])
    assert obj_exact_energy.weights["energy"] == pytest.approx(100.0)

