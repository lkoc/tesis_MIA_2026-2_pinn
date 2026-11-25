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


def test_divergence_guard_rules() -> None:
    from pinn_thermal.core.trainer import DivergenceGuard

    # 1. NaN/Inf y max_loss inmediatos
    g1 = DivergenceGuard({"max_loss": 1e6})
    assert g1.check_step(3, "adam", float("nan")) is True
    assert g1.reason == "non_finite_loss"

    g2 = DivergenceGuard({"max_loss": 1e6})
    assert g2.check_step(5, "adam", 2e6) is True
    assert g2.reason == "loss_exceeded_max_loss"

    # 2. Explosión relativa con paciencia (no dispara en un pico aislado tras resample)
    g3 = DivergenceGuard({"warmup_steps": 10, "explosion_factor": 100.0, "patience_explosion": 3})
    assert g3.check_adam_log(10, 1.0) is False
    assert g3.check_adam_log(20, 200.0) is False
    g3.notify_resample()
    assert g3.check_adam_log(30, 50.0) is False  # resetea racha
    assert g3.check_adam_log(40, 200.0) is False
    assert g3.check_adam_log(50, 300.0) is False
    assert g3.check_adam_log(60, 400.0) is True
    assert g3.reason == "loss_explosion_vs_best"

    # 3. Crecimiento sostenido
    g4 = DivergenceGuard({"warmup_steps": 10, "growth_patience": 3, "min_growth_ratio": 1.05})
    for step, loss_v in ((10, 1.0), (20, 1.2), (30, 1.4), (40, 1.6)):
        res = g4.check_adam_log(step, loss_v)
    assert res is True
    assert g4.reason == "sustained_loss_growth"

    # 4. Meseta temprana en L-BFGS
    g5 = DivergenceGuard({"lbfgs_stagnation_checks": 3, "lbfgs_min_rel_improve": 1e-4})
    assert g5.check_lbfgs_log(100, 1.0) is None
    assert g5.check_lbfgs_log(101, 1.0) is None
    assert g5.check_lbfgs_log(102, 1.0) is None
    assert g5.check_lbfgs_log(103, 1.0) == "plateau"
    assert g5.lbfgs_early_stop is not None and g5.lbfgs_early_stop["reason"] == "lbfgs_plateau"


def test_trainer_and_parallel_runner_stop_on_divergence(tmp_path: Path, mms_problem: ThermalProblem2D) -> None:
    import json
    from pinn_thermal.core.trainer import PINNTrainer, TrainConfig

    # Corrida intencionalmente inestable (lr=1e3) debe abortar muy antes de 300 pasos y saltar L-BFGS
    cfg = TrainConfig(
        adam_steps=300,
        lbfgs_steps=20,
        lr=1e3,
        width=12,
        depth=2,
        log_every=10,
        device="cpu",
        dtype="float32",
        full_audit=False,
        divergence_stop={"max_loss": 1e6, "warmup_steps": 5, "growth_patience": 3},
    )
    art = PINNTrainer(cfg).train(mms_problem.doc, "M0", tmp_path / "div_direct", overwrite=True)
    assert art.result["status"] == "diverged"
    assert art.result["divergence_stop"]["triggered"] is True
    assert art.result["divergence_stop"]["step"] < 300
    assert art.result["training_elapsed_s"]["lbfgs"] == 0.0
    prog = json.loads((tmp_path / "div_direct" / "progress.json").read_text(encoding="utf-8"))
    assert prog["status"] == "diverged"

    # ParallelRunner con una tarea normal y una divergente
    tasks = [
        RunTask(
            name="ok_run",
            problem_source=mms_problem.doc,
            architecture="M1",
            output_dir=tmp_path / "par_ok",
            config_kwargs={"adam_steps": 15, "lbfgs_steps": 0, "width": 12, "depth": 2, "device": "cpu", "full_audit": False},
            overwrite=True,
        ),
        RunTask(
            name="div_run",
            problem_source=mms_problem.doc,
            architecture="M0",
            output_dir=tmp_path / "par_div",
            config_kwargs={
                "adam_steps": 200,
                "lbfgs_steps": 10,
                "lr": 1e3,
                "width": 12,
                "depth": 2,
                "log_every": 10,
                "device": "cpu",
                "full_audit": False,
                "divergence_stop": {"max_loss": 1e6, "warmup_steps": 5},
            },
            overwrite=True,
        ),
    ]
    results = ParallelRunner(max_workers=2).run(tasks, show_progress=True, progress_interval=0.2)
    assert results[0]["status"] == "completed"
    assert results[1]["status"] == "diverged"
    assert results[1]["divergence_stop"]["triggered"] is True
