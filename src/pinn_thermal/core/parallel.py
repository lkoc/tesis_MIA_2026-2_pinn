"""Ejecutor paralelo opcional para campañas multi-modelo, multi-caso y multi-semilla.

Utiliza procesos independientes con contexto ``spawn`` (compatible con Windows, Linux,
PyTorch CPU y aceleradores CUDA/XPU/NPU), asignando dispositivos por trabajador y
limitando los hilos CPU por proceso para evitar sobre-suscripción.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from dataclasses import dataclass, field
import html
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterator
import warnings

import numpy as np

from pinn_thermal.core.device import DeviceManager
from pinn_thermal.core.manifest import replace_json_atomic


def _ensure_spawn_pythonpath() -> None:
    """Garantiza que los procesos hijo ``spawn`` puedan importar ``pinn_thermal`` en Windows."""
    pkg_root = str(Path(__file__).resolve().parents[2])
    existing = os.environ.get("PYTHONPATH", "")
    parts = [p for p in existing.split(os.pathsep) if p] if existing else []
    if pkg_root not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([pkg_root, *parts]) if parts else pkg_root


@contextmanager
def _worker_thread_env(cpu_threads: int) -> Iterator[None]:
    """Limita los hilos OpenMP/BLAS antes de crear procesos ``spawn`` para evitar sobre-suscripción."""
    keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    prev = {k: os.environ.get(k) for k in keys}
    val = str(max(1, int(cpu_threads)))
    for k in keys:
        os.environ[k] = val
    try:
        yield
    finally:
        for k, old in prev.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


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


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "--:--"
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore[import-not-found]

        ip = get_ipython()
        return ip is not None and "IPKernelApp" in getattr(ip, "config", {})
    except Exception:
        return False


def _planned_steps_for_task(task: RunTask | AdaptiveRunTask) -> int:
    if isinstance(task, AdaptiveRunTask):
        return max(sum(int(x) for x in task.adam_blocks) + sum(int(x) for x in task.lbfgs_blocks), 1)
    cfg = task.config_kwargs
    adam = int(cfg.get("adam_steps", 1500))
    lbfgs = 0 if cfg.get("attention") is not None else int(cfg.get("lbfgs_steps", 100))
    return max(adam + lbfgs, 1)


def _progress_path_for_task(task: RunTask | AdaptiveRunTask) -> Path:
    if isinstance(task, AdaptiveRunTask):
        return Path(task.run_root) / "progress.json"
    return Path(task.output_dir) / "progress.json"


class _LiveProgressDisplay:
    """Monitor ligero de progreso en vivo (HTML en Jupyter/VS Code o texto en consola)."""

    def __init__(
        self,
        tasks: list[RunTask] | list[AdaptiveRunTask],
        *,
        enabled: bool = False,
        title: str = "Avance de campaña",
        workers: int = 1,
    ) -> None:
        self.enabled = bool(enabled)
        self.title = title
        self.workers = max(1, int(workers))
        self.started_mono = time.perf_counter()
        self.started_wall = time.time() - 0.05
        self._notebook = self.enabled and _in_notebook()
        self._display_handle: Any = None
        self._last_text_render = 0.0
        self.items: list[dict[str, Any]] = [
            {
                "name": t.name,
                "path": _progress_path_for_task(t),
                "total_steps": _planned_steps_for_task(t),
                "completed_steps": 0,
                "pct": 0.0,
                "status": "pending",
                "phase": "-",
                "loss": None,
                "elapsed_s": 0.0,
                "done": False,
            }
            for t in tasks
        ]

    def set_workers(self, workers: int) -> None:
        self.workers = max(1, int(workers))

    def mark_finished(self, idx: int, res: dict[str, Any]) -> None:
        item = self.items[idx]
        status = str(res.get("status", "completed"))
        item["done"] = True
        item["status"] = status
        if status == "completed":
            item["completed_steps"] = item["total_steps"]
            item["pct"] = 100.0
            item["phase"] = "done"
        else:
            item["phase"] = "failed"
        if res.get("elapsed_s") is not None:
            item["elapsed_s"] = float(res["elapsed_s"])
        if res.get("best_loss") is not None and np.isfinite(res["best_loss"]):
            item["loss"] = float(res["best_loss"])

    def poll_disk(self) -> None:
        for item in self.items:
            if item["done"]:
                continue
            path: Path = item["path"]
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_mtime < self.started_wall:
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            total = int(data.get("total_steps") or item["total_steps"])
            if "completed_steps" in data and data["completed_steps"] is not None:
                comp = int(data["completed_steps"])
            elif "step" in data and data["step"] is not None:
                comp = max(int(data["step"]) + 1, 0)
            else:
                comp = item["completed_steps"]
            comp = max(0, min(comp, total))
            item["total_steps"] = max(total, 1)
            item["completed_steps"] = comp
            item["pct"] = float(data.get("pct")) if data.get("pct") is not None else (100.0 * comp / item["total_steps"])
            item["status"] = str(data.get("status") or "running")
            item["phase"] = str(data.get("phase") or "running")
            if data.get("loss_total") is not None:
                try:
                    val = float(data["loss_total"])
                    if np.isfinite(val):
                        item["loss"] = val
                except (TypeError, ValueError):
                    pass
            if data.get("elapsed_s") is not None:
                try:
                    item["elapsed_s"] = float(data["elapsed_s"])
                except (TypeError, ValueError):
                    pass

    def _summary(self) -> dict[str, Any]:
        total_steps = sum(it["total_steps"] for it in self.items) or 1
        done_steps = sum(it["completed_steps"] for it in self.items)
        pct = min(100.0, 100.0 * done_steps / total_steps)
        elapsed = time.perf_counter() - self.started_mono
        eta = (elapsed * (100.0 - pct) / pct) if 0.5 <= pct < 100.0 else (0.0 if pct >= 100.0 else None)
        n_done = sum(1 for it in self.items if it["done"] and it["status"] == "completed")
        n_fail = sum(1 for it in self.items if it["done"] and it["status"] != "completed")
        n_run = sum(1 for it in self.items if not it["done"] and it["status"] == "running")
        n_pend = len(self.items) - n_done - n_fail - n_run
        return {
            "total_steps": total_steps,
            "done_steps": done_steps,
            "pct": pct,
            "elapsed": elapsed,
            "eta": eta,
            "n_total": len(self.items),
            "n_done": n_done,
            "n_fail": n_fail,
            "n_run": n_run,
            "n_pend": n_pend,
        }

    def _render_html(self, s: dict[str, Any]) -> str:
        # Muestra todas las tareas activas y hasta 24 filas totales para mantener el panel compacto
        running = [it for it in self.items if not it["done"] and it["status"] == "running"]
        finished = [it for it in self.items if it["done"]]
        pending = [it for it in self.items if not it["done"] and it["status"] != "running"]
        visible = self.items if len(self.items) <= 24 else (running + finished[-10:] + pending[:4])

        rows_html = []
        for it in visible:
            st = it["status"]
            badge_bg = {"completed": "#1f7a4c", "running": "#1d6fa5", "failed": "#b9382c"}.get(st, "#6c757d")
            loss_s = f"{it['loss']:.2e}" if it["loss"] is not None else "-"
            el_s = f"{it['elapsed_s']:.1f}s" if it["elapsed_s"] else "-"
            pct_v = max(0.0, min(100.0, float(it["pct"])))
            rows_html.append(
                "<tr>"
                f"<td style='padding:3px 8px;font-family:monospace;'>{html.escape(it['name'])}</td>"
                f"<td style='padding:3px 8px;'><span style='background:{badge_bg};color:#fff;padding:1px 6px;border-radius:4px;font-size:11px;'>{html.escape(st)}</span></td>"
                f"<td style='padding:3px 8px;font-family:monospace;'>{html.escape(it['phase'])}</td>"
                f"<td style='padding:3px 8px;text-align:right;font-family:monospace;'>{it['completed_steps']:,} / {it['total_steps']:,}</td>"
                f"<td style='padding:3px 8px;min-width:120px;'>"
                f"<div style='background:#e9ecef;border-radius:4px;height:10px;overflow:hidden;'>"
                f"<div style='width:{pct_v:.1f}%;background:{badge_bg};height:100%;'></div></div>"
                f"</td>"
                f"<td style='padding:3px 8px;text-align:right;font-family:monospace;'>{pct_v:5.1f}%</td>"
                f"<td style='padding:3px 8px;text-align:right;font-family:monospace;'>{loss_s}</td>"
                f"<td style='padding:3px 8px;text-align:right;font-family:monospace;'>{el_s}</td>"
                "</tr>"
            )
        omitted = len(self.items) - len(visible)
        extra_note = (
            f"<div style='font-size:11px;color:#666;margin-top:4px;'>… y {omitted} tarea(s) adicional(es) en cola/completadas.</div>"
            if omitted > 0
            else ""
        )
        fail_str = f" | Fallidas: <b>{s['n_fail']}</b>" if s["n_fail"] else ""
        return (
            "<div style='border:1px solid #d0d7de;border-radius:8px;padding:10px 12px;margin:4px 0;font-size:12px;'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;'>"
            f"<b>{html.escape(self.title)}</b>"
            f"<span style='font-family:monospace;'><b>{s['pct']:5.1f}%</b> | Pasos: {s['done_steps']:,} / {s['total_steps']:,} "
            f"| Completadas: <b>{s['n_done']}/{s['n_total']}</b>{fail_str} | En curso: {s['n_run']} (workers={self.workers}) "
            f"| Tiempo: {_format_duration(s['elapsed'])} | ETA: {_format_duration(s['eta'])}</span>"
            "</div>"
            "<div style='background:#e9ecef;border-radius:6px;height:14px;overflow:hidden;margin-bottom:8px;'>"
            f"<div style='width:{s['pct']:.1f}%;background:#1f7a4c;height:100%;transition:width 0.25s;'></div>"
            "</div>"
            "<table style='width:100%;border-collapse:collapse;font-size:12px;'>"
            "<thead><tr style='border-bottom:1px solid #d0d7de;text-align:left;'>"
            "<th style='padding:3px 8px;'>Tarea</th><th style='padding:3px 8px;'>Estado</th>"
            "<th style='padding:3px 8px;'>Fase</th><th style='padding:3px 8px;text-align:right;'>Pasos</th>"
            "<th style='padding:3px 8px;'>Progreso</th><th style='padding:3px 8px;text-align:right;'>%</th>"
            "<th style='padding:3px 8px;text-align:right;'>Loss</th><th style='padding:3px 8px;text-align:right;'>Tiempo</th>"
            "</tr></thead>"
            f"<tbody>{''.join(rows_html)}</tbody></table>"
            f"{extra_note}"
            "</div>"
        )

    def render(self, *, force: bool = False) -> None:
        if not self.enabled:
            return
        self.poll_disk()
        s = self._summary()
        if self._notebook:
            from IPython.display import HTML, display  # type: ignore[import-not-found]

            html_obj = HTML(self._render_html(s))
            if self._display_handle is None:
                self._display_handle = display(html_obj, display_id=True)
            else:
                self._display_handle.update(html_obj)
            return

        now = time.perf_counter()
        if not force and (now - self._last_text_render) < 1.0:
            return
        self._last_text_render = now
        active = [
            f"{it['name']}:{it['phase']}({it['completed_steps']}/{it['total_steps']}={it['pct']:.0f}%)"
            for it in self.items
            if not it["done"] and it["status"] == "running"
        ][:4]
        active_str = (" | " + ", ".join(active)) if active else ""
        msg = (
            f"[{self.title}] {s['pct']:5.1f}% | pasos {s['done_steps']:,}/{s['total_steps']:,} "
            f"| tareas {s['n_done']}/{s['n_total']} | t={_format_duration(s['elapsed'])} "
            f"| ETA={_format_duration(s['eta'])}{active_str}"
        )
        print(msg, file=sys.stdout, flush=True)


def _worker_execute(
    task: RunTask,
    cpu_threads: int,
    on_log: Callable[[dict[str, Any], Any], None] | None = None,
) -> dict[str, Any]:
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
        on_log=on_log,
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


def _adaptive_worker_execute(
    task: AdaptiveRunTask,
    cpu_threads: int,
    on_log: Callable[[dict[str, Any], Any], None] | None = None,
) -> dict[str, Any]:
    """Ejecuta de forma aislada una cadena adaptativa completa ``adam_i -> lbfgs_j``."""
    import pinn_thermal.models  # noqa: F401 - registra las arquitecturas disponibles
    from pinn_thermal.core.trainer import PINNTrainer, TrainConfig

    DeviceManager.configure_threads(cpu_threads)
    started = time.perf_counter()
    run_root = Path(task.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    total_steps = _planned_steps_for_task(task)
    completed_before = 0
    stages: list[dict[str, Any]] = []
    prev_ckpt: Path | None = None
    relaxed_hit = task.reference is None
    strict_hit = False

    def _emit_progress(phase_label: str, stage_step: int, loss_val: float | None, status: str = "running") -> None:
        comp = min(completed_before + max(stage_step, 0), total_steps)
        pct = 100.0 if status == "completed" else round(100.0 * comp / total_steps, 2)
        replace_json_atomic(
            run_root / "progress.json",
            {
                "status": status,
                "phase": phase_label,
                "step": comp,
                "completed_steps": total_steps if status == "completed" else comp,
                "total_steps": total_steps,
                "pct": pct,
                "loss_total": loss_val,
                "elapsed_s": round(time.perf_counter() - started, 2),
            },
        )

    _emit_progress("init", 0, None)

    for idx, b_steps in enumerate(task.adam_blocks, 1):
        if task.reference is not None and relaxed_hit:
            break
        stage_name = f"adam_{idx}"
        s_dir = run_root / stage_name
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

        def _stage_log(row: dict[str, Any], _model: Any, _name: str = stage_name) -> None:
            _emit_progress(_name, int(row["step"]) + 1, float(row["total"]))
            if on_log is not None:
                on_log(row, _model)

        art = PINNTrainer(TrainConfig(**cfg_kwargs)).train(
            problem_source=task.problem_source,
            architecture=task.architecture,
            output_dir=s_dir,
            reference=task.reference,
            reference_label=task.reference_label,
            on_log=_stage_log,
            overwrite=task.overwrite,
        )
        completed_before += int(b_steps)
        prev_ckpt = s_dir / "model.pt"
        m = _extract_stage_metrics(art.result)
        stages.append({"name": stage_name, "steps": int(b_steps), "metrics": m, "status": art.result.get("status")})
        if task.reference is not None and _passes_criteria(m, task.relaxed_criteria):
            relaxed_hit = True
            break

    for idx, l_steps in enumerate(task.lbfgs_blocks, 1):
        stage_name = f"lbfgs_{idx}"
        s_dir = run_root / stage_name
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

        def _stage_log_lbfgs(row: dict[str, Any], _model: Any, _name: str = stage_name) -> None:
            _emit_progress(_name, int(row["step"]) + 1, float(row["total"]))
            if on_log is not None:
                on_log(row, _model)

        art = PINNTrainer(TrainConfig(**cfg_kwargs)).train(
            problem_source=task.problem_source,
            architecture=task.architecture,
            output_dir=s_dir,
            reference=task.reference,
            reference_label=task.reference_label,
            on_log=_stage_log_lbfgs,
            overwrite=task.overwrite,
        )
        completed_before += int(l_steps)
        prev_ckpt = s_dir / "model.pt"
        m = _extract_stage_metrics(art.result)
        stages.append({"name": stage_name, "steps": int(l_steps), "metrics": m, "status": art.result.get("status")})
        if task.reference is not None and _passes_criteria(m, task.strict_criteria):
            strict_hit = True
            break

    _emit_progress("done", total_steps, None, status="completed")
    return {
        "name": task.name,
        "case": task.case_id,
        "architecture": task.architecture,
        "seed": int(task.seed),
        "status": "completed",
        "relaxed_reached": relaxed_hit,
        "strict_converged": strict_hit,
        "elapsed_s": time.perf_counter() - started,
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
        show_progress: bool = False,
        progress_title: str = "Avance de campaña",
        progress_interval: float = 1.0,
    ) -> list[dict[str, Any]]:
        """Alias de :meth:`run` para compatibilidad con invocaciones de lote."""
        return self.run(
            tasks,
            on_complete=on_complete,
            show_progress=show_progress,
            progress_title=progress_title,
            progress_interval=progress_interval,
        )

    def run(
        self,
        tasks: list[RunTask],
        on_complete: Callable[[dict[str, Any]], None] | None = None,
        show_progress: bool = False,
        progress_title: str = "Avance de campaña",
        progress_interval: float = 1.0,
    ) -> list[dict[str, Any]]:
        """Ejecuta todas las tareas y devuelve sus resúmenes en el mismo orden de entrada."""
        if not tasks:
            return []
        total_cpus = os.cpu_count() or 1
        ordered: dict[int, dict[str, Any]] = {}
        pending: list[int] = list(range(len(tasks)))
        workers = min(self.max_workers, len(pending))
        poll_s = max(0.2, float(progress_interval))
        tracker = _LiveProgressDisplay(tasks, enabled=show_progress, title=progress_title, workers=workers)
        tracker.render(force=True)

        while pending:
            workers = min(workers, len(pending))
            tracker.set_workers(workers)
            cpu_threads = max(1, total_cpus // workers)

            if workers == 1:
                for idx in pending:
                    task = tasks[idx]
                    assigned = self._assign_task_device(task, idx)
                    log_cb = (lambda _row, _m: tracker.render()) if show_progress else None
                    try:
                        res = _worker_execute(assigned, cpu_threads, on_log=log_cb)
                    except Exception as exc:
                        res = {
                            "name": task.name,
                            "status": "failed",
                            "output_dir": str(task.output_dir),
                            "architecture": task.architecture,
                            "error": {"type": type(exc).__name__, "message": str(exc)},
                        }
                    ordered[idx] = res
                    tracker.mark_finished(idx, res)
                    tracker.render(force=True)
                    if on_complete is not None:
                        on_complete(res)
                break

            _ensure_spawn_pythonpath()
            ctx = multiprocessing.get_context("spawn")
            pool_broken = False
            try:
                with _worker_thread_env(cpu_threads), ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
                    future_map = {}
                    for idx in pending:
                        task = tasks[idx]
                        assigned = self._assign_task_device(task, idx)
                        future = pool.submit(_worker_execute, assigned, cpu_threads)
                        future_map[future] = (idx, task)
                    unfinished = set(future_map)
                    while unfinished:
                        done_set, unfinished = wait(
                            unfinished,
                            timeout=poll_s if show_progress else None,
                            return_when=FIRST_COMPLETED,
                        )
                        for future in done_set:
                            idx, task = future_map[future]
                            try:
                                res = future.result()
                            except BrokenProcessPool:
                                pool_broken = True
                                break
                            except Exception as exc:
                                res = {
                                    "name": task.name,
                                    "status": "failed",
                                    "output_dir": str(task.output_dir),
                                    "architecture": task.architecture,
                                    "error": {"type": type(exc).__name__, "message": str(exc)},
                                }
                            ordered[idx] = res
                            tracker.mark_finished(idx, res)
                            if on_complete is not None:
                                on_complete(res)
                        if pool_broken:
                            break
                        tracker.render(force=bool(done_set))
            except BrokenProcessPool:
                pool_broken = True

            pending = [idx for idx in range(len(tasks)) if idx not in ordered]
            if pending and pool_broken:
                next_workers = max(1, workers // 2)
                warnings.warn(
                    f"El pool de {workers} procesos se interrumpió (BrokenProcessPool); "
                    f"reintentando {len(pending)} tarea(s) pendientes con {next_workers} trabajador(es).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                workers = next_workers
        tracker.render(force=True)
        return [ordered[i] for i in range(len(tasks))]

    def run_adaptive(
        self,
        tasks: list[AdaptiveRunTask],
        on_complete: Callable[[dict[str, Any]], None] | None = None,
        show_progress: bool = False,
        progress_title: str = "Avance adaptativo",
        progress_interval: float = 1.0,
    ) -> list[dict[str, Any]]:
        """Ejecuta cadenas adaptativas ``adam_i -> lbfgs_j`` en paralelo con contexto ``spawn``."""
        if not tasks:
            return []
        total_cpus = os.cpu_count() or 1
        ordered: dict[int, dict[str, Any]] = {}
        pending: list[int] = list(range(len(tasks)))
        workers = min(self.max_workers, len(pending))
        poll_s = max(0.2, float(progress_interval))
        tracker = _LiveProgressDisplay(tasks, enabled=show_progress, title=progress_title, workers=workers)
        tracker.render(force=True)

        while pending:
            workers = min(workers, len(pending))
            tracker.set_workers(workers)
            cpu_threads = max(1, total_cpus // workers)

            if workers == 1:
                for idx in pending:
                    task = tasks[idx]
                    log_cb = (lambda _row, _m: tracker.render()) if show_progress else None
                    try:
                        res = _adaptive_worker_execute(task, cpu_threads, on_log=log_cb)
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
                    tracker.mark_finished(idx, res)
                    tracker.render(force=True)
                    if on_complete is not None:
                        on_complete(res)
                break

            _ensure_spawn_pythonpath()
            ctx = multiprocessing.get_context("spawn")
            pool_broken = False
            try:
                with _worker_thread_env(cpu_threads), ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
                    future_map = {
                        pool.submit(_adaptive_worker_execute, tasks[idx], cpu_threads): (idx, tasks[idx])
                        for idx in pending
                    }
                    unfinished = set(future_map)
                    while unfinished:
                        done_set, unfinished = wait(
                            unfinished,
                            timeout=poll_s if show_progress else None,
                            return_when=FIRST_COMPLETED,
                        )
                        for future in done_set:
                            idx, task = future_map[future]
                            try:
                                res = future.result()
                            except BrokenProcessPool:
                                pool_broken = True
                                break
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
                            tracker.mark_finished(idx, res)
                            if on_complete is not None:
                                on_complete(res)
                        if pool_broken:
                            break
                        tracker.render(force=bool(done_set))
            except BrokenProcessPool:
                pool_broken = True

            pending = [idx for idx in range(len(tasks)) if idx not in ordered]
            if pending and pool_broken:
                next_workers = max(1, workers // 2)
                warnings.warn(
                    f"El pool adaptativo de {workers} procesos se interrumpió (BrokenProcessPool); "
                    f"reintentando {len(pending)} tarea(s) pendientes con {next_workers} trabajador(es).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                workers = next_workers
        tracker.render(force=True)
        return [ordered[i] for i in range(len(tasks))]
