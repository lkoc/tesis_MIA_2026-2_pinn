"""Pruebas de selección transparente de dispositivo y paralelismo multi-proceso."""

from __future__ import annotations

from pathlib import Path

import torch

from pinn_thermal.core.device import DeviceManager
from pinn_thermal.core.parallel import ParallelRunner, RunTask
from pinn_thermal.core.problem import ThermalProblem2D


def test_device_manager_and_dtype_resolution() -> None:
    desc = DeviceManager.describe()
    assert "cpu" in desc["backends"]
    dev = DeviceManager.resolve("cpu")
    assert dev.type == "cpu"
    assert DeviceManager.resolve_dtype(dev, "float32") == torch.float32
    assert DeviceManager.resolve_dtype(dev, "float64") == torch.float64

    # Verifica captura/restauración determinista RNG
    DeviceManager.seed_all(42, dev)
    snap = DeviceManager.capture_rng(dev)
    r1 = torch.rand(4)
    DeviceManager.restore_rng(snap)
    r2 = torch.rand(4)
    assert torch.allclose(r1, r2)


def test_parallel_runner_executes_spawn_pool(tmp_path: Path, mms_problem: ThermalProblem2D) -> None:
    prob_path = tmp_path / "problem.json"
    from pinn_thermal.core.manifest import write_json_atomic

    write_json_atomic(prob_path, mms_problem.doc)
    tasks = [
        RunTask(
            name="t_m1",
            problem_source=str(prob_path),
            architecture="M1",
            output_dir=str(tmp_path / "out_m1"),
            config_kwargs={"adam_steps": 15, "lbfgs_steps": 0, "width": 12, "depth": 2, "device": "cpu"},
            overwrite=True,
        ),
        RunTask(
            name="t_cuspr",
            problem_source=str(prob_path),
            architecture="CUSPR",
            output_dir=str(tmp_path / "out_cuspr"),
            config_kwargs={"adam_steps": 15, "lbfgs_steps": 0, "width": 12, "depth": 2, "device": "cpu"},
            overwrite=True,
        ),
    ]
    results = ParallelRunner(max_workers=2).run(tasks)
    assert len(results) == 2
    assert all(r["status"] == "completed" for r in results)
    assert (tmp_path / "out_m1" / "model.pt").is_file()
    assert (tmp_path / "out_cuspr" / "model.pt").is_file()
