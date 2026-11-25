"""Pruebas de selección transparente de dispositivo y paralelismo multi-proceso."""

from __future__ import annotations

import os
from pathlib import Path
import warnings

import torch

from pinn_thermal.core.device import DeviceManager
from pinn_thermal.core.parallel import AdaptiveRunTask, ParallelRunner, RunTask
from pinn_thermal.core.problem import ThermalProblem2D


def test_device_manager_and_dtype_resolution() -> None:
    desc = DeviceManager.describe()
    assert "cpu" in desc["backends"]
    dev = DeviceManager.resolve("cpu")
    assert dev.type == "cpu"
    assert DeviceManager.resolve_dtype(dev, "float32") == torch.float32
    assert DeviceManager.resolve_dtype(dev, "float64") == torch.float64

    # Verifica que torch.dot / Tensor.dot sigan intactos en CPU
    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    b = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32)
    assert torch.allclose(a.dot(b), torch.tensor(32.0))
    assert torch.allclose(torch.dot(a, b), torch.tensor(32.0))

    # Verifica captura/restauración determinista RNG
    DeviceManager.seed_all(42, dev)
    snap = DeviceManager.capture_rng(dev)
    r1 = torch.rand(4)
    DeviceManager.restore_rng(snap)
    r2 = torch.rand(4)
    assert torch.allclose(r1, r2)


def test_xpu_native_dot_and_lbfgs_no_cpu_fallback() -> None:
    """Si Intel XPU está disponible, verifica 0 fallbacks a CPU en dot() y LBFGS."""
    info = DeviceManager.available_backends()
    if not info["xpu"]["available"]:
        return

    assert DeviceManager.patch_xpu_ops() is True
    dev = DeviceManager.resolve("xpu")
    prev = os.environ.get("PYTORCH_DEBUG_XPU_FALLBACK")
    os.environ["PYTORCH_DEBUG_XPU_FALLBACK"] = "1"
    try:
        a_cpu = torch.randn(128, dtype=torch.float32)
        b_cpu = torch.randn(128, dtype=torch.float32)
        expected = a_cpu.dot(b_cpu)
        a_xpu = a_cpu.to(dev)
        b_xpu = b_cpu.to(dev)
        out_xpu = torch.empty((), device=dev, dtype=torch.float32)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r1 = a_xpu.dot(b_xpu)
            r2 = torch.dot(a_xpu, b_xpu)
            r3 = torch.dot(a_xpu, b_xpu, out=out_xpu)

            p = torch.nn.Parameter(torch.randn(32, device=dev, dtype=torch.float32))
            opt = torch.optim.LBFGS([p], max_iter=5, history_size=10, line_search_fn="strong_wolfe")

            def closure() -> torch.Tensor:
                opt.zero_grad()
                loss = (p ** 2).sum()
                loss.backward()
                return loss

            opt.step(closure)

        fallback_warnings = [str(w.message) for w in caught if "fallback" in str(w.message).lower() or "falling back" in str(w.message).lower()]
        assert not fallback_warnings, f"Se detectaron fallbacks XPU->CPU inesperados: {fallback_warnings}"
        assert r1.device.type == "xpu" and r2.device.type == "xpu" and r3.device.type == "xpu"
        assert torch.allclose(r1.cpu(), expected, atol=1e-4)
        assert torch.allclose(r2.cpu(), expected, atol=1e-4)
        assert torch.allclose(out_xpu.cpu(), expected, atol=1e-4)
    finally:
        if prev is None:
            os.environ.pop("PYTORCH_DEBUG_XPU_FALLBACK", None)
        else:
            os.environ["PYTORCH_DEBUG_XPU_FALLBACK"] = prev


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
    completed: list[str] = []
    results = ParallelRunner(max_workers=2).run(
        tasks,
        on_complete=lambda r: completed.append(r["name"]),
        show_progress=True,
        progress_interval=0.25,
    )
    assert len(results) == 2
    assert set(completed) == {"t_m1", "t_cuspr"}
    assert all(r["status"] == "completed" for r in results)
    assert all(r["device"] == "cpu" for r in results)
    assert all("best_loss" in r for r in results)
    assert (tmp_path / "out_m1" / "model.pt").is_file()
    assert (tmp_path / "out_cuspr" / "model.pt").is_file()

    import json

    prog = json.loads((tmp_path / "out_m1" / "progress.json").read_text(encoding="utf-8"))
    assert prog["status"] == "completed"
    assert prog["total_steps"] == 15
    assert prog["completed_steps"] == 15
    assert prog["pct"] == 100.0


def test_parallel_runner_executes_adaptive_tasks(tmp_path: Path, mms_problem: ThermalProblem2D) -> None:
    tasks = [
        AdaptiveRunTask(
            name="a_m0",
            case_id=mms_problem.id,
            problem_source=mms_problem.doc,
            architecture="M0",
            seed=0,
            run_root=tmp_path / "adapt_m0",
            adam_blocks=(8,),
            lbfgs_blocks=(2,),
            relaxed_criteria={"error_Tmax_abs_K": 0.1, "nrmse_pct": 0.5, "balance_pct": 1.0},
            strict_criteria={"error_Tmax_abs_K": 0.05, "nrmse_pct": 0.25, "balance_pct": 0.5},
            base_config_kwargs={"lr": 2e-3, "width": 12, "depth": 2, "dtype": "float32", "device": "cpu"},
            overwrite=True,
        ),
        AdaptiveRunTask(
            name="a_m1",
            case_id=mms_problem.id,
            problem_source=mms_problem.doc,
            architecture="M1",
            seed=0,
            run_root=tmp_path / "adapt_m1",
            adam_blocks=(8,),
            lbfgs_blocks=(2,),
            relaxed_criteria={"error_Tmax_abs_K": 0.1, "nrmse_pct": 0.5, "balance_pct": 1.0},
            strict_criteria={"error_Tmax_abs_K": 0.05, "nrmse_pct": 0.25, "balance_pct": 0.5},
            base_config_kwargs={"lr": 2e-3, "width": 12, "depth": 2, "dtype": "float32", "device": "cpu"},
            overwrite=True,
        ),
    ]
    results = ParallelRunner(max_workers=2).run_adaptive(tasks)
    assert len(results) == 2
    assert all(r["status"] == "completed" for r in results)
    assert all(len(r["stages"]) == 2 for r in results)
