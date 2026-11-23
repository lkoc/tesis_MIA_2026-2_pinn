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
from typing import Any

from pinn_thermal.core.device import DeviceManager


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
        "elapsed_s": artifact.result.get("elapsed_s"),
        "evaluation": artifact.result.get("evaluation", {}),
        "error": artifact.result.get("error"),
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

    def run_batch(self, tasks: list[RunTask]) -> list[dict[str, Any]]:
        """Alias de :meth:`run` para compatibilidad con invocaciones de lote."""
        return self.run(tasks)

    def run(self, tasks: list[RunTask]) -> list[dict[str, Any]]:
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
                    results.append(_worker_execute(assigned, cpu_threads))
                except Exception as exc:
                    results.append({"name": task.name, "status": "failed", "error": {"type": type(exc).__name__, "message": str(exc)}})
            return results

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
                    ordered[idx] = future.result()
                except Exception as exc:
                    ordered[idx] = {
                        "name": task.name,
                        "status": "failed",
                        "output_dir": str(task.output_dir),
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    }
        return [ordered[i] for i in range(len(tasks))]
