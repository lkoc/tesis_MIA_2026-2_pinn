"""Ejecutor paralelo opcional para campañas multi-modelo, multi-caso y multi-semilla.

Utiliza procesos independientes con contexto ``spawn`` (compatible con Windows, Linux,
PyTorch CPU y aceleradores CUDA/XPU/NPU), asignando dispositivos por trabajador y
limitando los hilos CPU por proceso para evitar sobre-suscripción.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import multiprocessing
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

from pinn_thermal.core.device import DeviceManager


def _ensure_spawn_pythonpath() -> None:
    """Garantiza que los procesos hijo ``spawn`` puedan importar ``pinn_thermal`` en Windows."""
    pkg_root = str(Path(__file__).resolve().parents[2])
    existing = os.environ.get("PYTHONPATH", "")
    parts = [p for p in existing.split(os.pathsep) if p] if existing else []
    if pkg_root not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([pkg_root, *parts]) if parts else pkg_root


@dataclass
class RunTask:
    """Especificación autocontenida de una corrida de entrenamiento independiente."""

    name: str
    problem_source: str | Path | dict[str, Any]
    architecture: str
    output_dir: str | Path
    config_kwargs: dict[str, Any] = field(default_factory=dict)
    reference: Any = None
    reference_label: str = "supplied"
    overwrite: bool = False


@dataclass
class AdaptiveRunTask:
    """Especificación autocontenida de una cadena adaptativa ``adam_i -> lbfgs_j``."""

    name: str
    case_id: str
    problem_source: str | Path | dict[str, Any]
    architecture: str
    seed: int
    run_root: str | Path
    adam_blocks: tuple[int, ...]
    lbfgs_blocks: tuple[int, ...]
    relaxed_criteria: dict[str, float]
    strict_criteria: dict[str, float]
    base_config_kwargs: dict[str, Any] = field(default_factory=dict)
    reference: Any = None
    reference_label: str = "supplied"
    overwrite: bool = True


def _extract_stage_metrics(result: dict[str, Any]) -> dict[str, Any]:
    ev = result.get("evaluation") or {}
    bal = ev.get("balance") or {}
    return {
        "error_Tmax_abs_K": ev.get("error_Tmax_abs_K"),
        "nrmse_pct": ev.get("nrmse_pct"),
        "balance_pct": bal.get("global_derived_pct"),
        "Tmax_pred_C": ev.get("Tmax_pred_C"),
        "elapsed_s": result.get("elapsed_s"),
    }


def _passes_criteria(metrics: dict[str, Any], gate: dict[str, float]) -> bool:
    return all(
        metrics.get(k) is not None and np.isfinite(metrics[k]) and float(metrics[k]) < float(lim)
        for k, lim in gate.items()
    )


def _worker_execute(task: RunTask, cpu_threads: int) -> dict[str, Any]:
    """Función de nivel superior serializable por ``spawn`` para entrenar una tarea aislada."""
    # Importaciones diferidas dentro del proceso hijo
    import pinn_thermal.models  # noqa: F401 - registra las arquitecturas disponibles
    from pinn_thermal.core.trainer import PINNTrainer, TrainConfig

    DeviceManager.configure_threads(cpu_threads)
    cfg_kwargs = dict(task.config_kwargs)
    cfg = TrainConfig(**cfg_kwargs)
    trainer = PINNTrainer(config=cfg)
    artifact = trainer.train(
        problem_source=task.problem_source,
        architecture=task.architecture,
        output_dir=task.output_dir,
        reference=task.reference,
        reference_label=task.reference_label,
        overwrite=task.overwrite,
    )
    return {
        "name": task.name,
        "status": artifact.result.get("status", "completed"),
        "output_dir": str(Path(task.output_dir)),
        "architecture": task.architecture,
        "case": artifact.problem.id,
        "device": artifact.result.get("device"),
        "parameters": artifact.result.get("parameters"),
        "best_loss": artifact.result.get("best_loss"),
        "accuracy_stop": artifact.result.get("accuracy_stop"),
        "elapsed_s": artifact.result.get("elapsed_s"),
        "evaluation": artifact.result.get("evaluation", {}),
        "error": artifact.result.get("error"),
    }


def _adaptive_worker_execute(task: AdaptiveRunTask, cpu_threads: int) -> dict[str, Any]:
    """Ejecuta de forma aislada una cadena adaptativa completa ``adam_i -> lbfgs_j``."""
    import pinn_thermal.models  # noqa: F401 - registra las arquitecturas disponibles
    from pinn_thermal.core.trainer import PINNTrainer, TrainConfig

    DeviceManager.configure_threads(cpu_threads)
    run_root = Path(task.run_root)
    stages: list[dict[str, Any]] = []
    prev_ckpt: Path | None = None
    relaxed_hit = task.reference is None
    strict_hit = False

    for idx, b_steps in enumerate(task.adam_blocks, 1):
        if task.reference is not None and relaxed_hit:
            break
        s_dir = run_root / f"adam_{idx}"
        cfg_kwargs = dict(task.base_config_kwargs)
        cfg_kwargs.update(
            {
                "adam_steps": int(b_steps),
                "lbfgs_steps": 0,
                "init_seed": int(task.seed),
                "sampling_seed": 11 + int(task.seed),
                "resume_from": prev_ckpt,
            }
        )
        art = PINNTrainer(TrainConfig(**cfg_kwargs)).train(
            problem_source=task.problem_source,
            architecture=task.architecture,
            output_dir=s_dir,
            reference=task.reference,
            reference_label=task.reference_label,
            overwrite=task.overwrite,
        )
        prev_ckpt = s_dir / "model.pt"
        m = _extract_stage_metrics(art.result)
        stages.append({"name": f"adam_{idx}", "steps": int(b_steps), "metrics": m, "status": art.result.get("status")})
        if task.reference is not None and _passes_criteria(m, task.relaxed_criteria):
            relaxed_hit = True
            break

    for idx, l_steps in enumerate(task.lbfgs_blocks, 1):
        s_dir = run_root / f"lbfgs_{idx}"
        cfg_kwargs = dict(task.base_config_kwargs)
        cfg_kwargs.update(
            {
                "adam_steps": 0,
                "lbfgs_steps": int(l_steps),
                "init_seed": int(task.seed),
                "sampling_seed": 11 + int(task.seed),
                "resume_from": prev_ckpt,
            }
        )
        art = PINNTrainer(TrainConfig(**cfg_kwargs)).train(
            problem_source=task.problem_source,
            architecture=task.architecture,
            output_dir=s_dir,
            reference=task.reference,
            reference_label=task.reference_label,
            overwrite=task.overwrite,
        )
        prev_ckpt = s_dir / "model.pt"
        m = _extract_stage_metrics(art.result)
        stages.append({"name": f"lbfgs_{idx}", "steps": int(l_steps), "metrics": m, "status": art.result.get("status")})
        if task.reference is not None and _passes_criteria(m, task.strict_criteria):
            strict_hit = True
            break

    return {
        "name": task.name,
        "case": task.case_id,
        "architecture": task.architecture,
        "seed": int(task.seed),
        "status": "completed",
        "relaxed_reached": relaxed_hit,
        "strict_converged": strict_hit,
        "stages": stages,
    }


class ParallelRunner:
    """Orquestador de ejecución secuencial o paralela para conjuntos de ``RunTask``."""

    def __init__(self, max_workers: int | None = None, distribute_devices: bool = True) -> None:
        self.max_workers = max(1, int(max_workers or min(4, os.cpu_count() or 1)))
        self.distribute_devices = bool(distribute_devices)

    def _assign_task_device(self, task: RunTask, worker_index: int) -> RunTask:
        """Asigna dispositivos distintos en round-robin si existen múltiples GPUs."""
        kwargs = dict(task.config_kwargs)
        req_dev = kwargs.get("device", "auto")
        if self.distribute_devices and str(req_dev).lower() in ("auto", "gpu"):
            info = DeviceManager.available_backends()
            if info["cuda"]["available"] and info["cuda"]["count"] > 1:
                kwargs["device"] = f"cuda:{worker_index % info['cuda']['count']}"
            elif info["xpu"]["available"] and info["xpu"]["count"] > 1:
                kwargs["device"] = f"xpu:{worker_index % info['xpu']['count']}"
        return RunTask(
            name=task.name,
            problem_source=task.problem_source,
            architecture=task.architecture,
            output_dir=task.output_dir,
            config_kwargs=kwargs,
            reference=task.reference,
            reference_label=task.reference_label,
            overwrite=task.overwrite,
        )

    def run_batch(
        self,
        tasks: list[RunTask],
        on_complete: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Alias de :meth:`run` para compatibilidad con invocaciones de lote."""
        return self.run(tasks, on_complete=on_complete)

    def run(
        self,
        tasks: list[RunTask],
        on_complete: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Ejecuta todas las tareas y devuelve sus resúmenes en el mismo orden de entrada."""
        if not tasks:
            return []
        workers = min(self.max_workers, len(tasks))
        total_cpus = os.cpu_count() or 1
        cpu_threads = max(1, total_cpus // workers)

        if workers == 1:
            results = []
            for idx, task in enumerate(tasks):
                assigned = self._assign_task_device(task, idx)
                try:
                    res = _worker_execute(assigned, cpu_threads)
                except Exception as exc:
                    res = {
                        "name": task.name,
                        "status": "failed",
                        "output_dir": str(task.output_dir),
                        "architecture": task.architecture,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    }
                results.append(res)
                if on_complete is not None:
                    on_complete(res)
            return results

        _ensure_spawn_pythonpath()
        ctx = multiprocessing.get_context("spawn")
        ordered: dict[int, dict[str, Any]] = {}
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            future_map = {}
            for idx, task in enumerate(tasks):
                assigned = self._assign_task_device(task, idx)
                future = pool.submit(_worker_execute, assigned, cpu_threads)
                future_map[future] = (idx, task)
            for future in as_completed(future_map):
                idx, task = future_map[future]
                try:
                    res = future.result()
                except Exception as exc:
                    res = {
                        "name": task.name,
                        "status": "failed",
                        "output_dir": str(task.output_dir),
                        "architecture": task.architecture,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    }
                ordered[idx] = res
                if on_complete is not None:
                    on_complete(res)
        return [ordered[i] for i in range(len(tasks))]

    def run_adaptive(
        self,
        tasks: list[AdaptiveRunTask],
        on_complete: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Ejecuta cadenas adaptativas ``adam_i -> lbfgs_j`` en paralelo con contexto ``spawn``."""
        if not tasks:
            return []
        workers = min(self.max_workers, len(tasks))
        total_cpus = os.cpu_count() or 1
        cpu_threads = max(1, total_cpus // workers)

        if workers == 1:
            results = []
            for task in tasks:
                try:
                    res = _adaptive_worker_execute(task, cpu_threads)
                except Exception as exc:
                    res = {
                        "name": task.name,
                        "case": task.case_id,
                        "architecture": task.architecture,
                        "seed": int(task.seed),
                        "status": "failed",
                        "relaxed_reached": False,
                        "strict_converged": False,
                        "stages": [],
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    }
                results.append(res)
                if on_complete is not None:
                    on_complete(res)
            return results

        _ensure_spawn_pythonpath()
        ctx = multiprocessing.get_context("spawn")
        ordered: dict[int, dict[str, Any]] = {}
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            future_map = {pool.submit(_adaptive_worker_execute, task, cpu_threads): (idx, task) for idx, task in enumerate(tasks)}
            for future in as_completed(future_map):
                idx, task = future_map[future]
                try:
                    res = future.result()
                except Exception as exc:
                    res = {
                        "name": task.name,
                        "case": task.case_id,
                        "architecture": task.architecture,
                        "seed": int(task.seed),
                        "status": "failed",
                        "relaxed_reached": False,
                        "strict_converged": False,
                        "stages": [],
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    }
                ordered[idx] = res
                if on_complete is not None:
                    on_complete(res)
        return [ordered[i] for i in range(len(tasks))]
