"""Entrenador principal unificado para todos los modelos PINN (``PINNTrainer``).

Soporta:
- Optimización en dos fases ``Adam + CosineAnnealingLR -> L-BFGS (strong_wolfe)``.
- Ejecución transparente en CPU, NVIDIA CUDA, AMD ROCm, Intel XPU, NPU, MPS y DirectML.
- Muestreo focalizado periódico, currículum progresivo de puntos, atención RBA/SA y parada temprana.
- Continuación mediante cualquiera de los cuatro mecanismos de :mod:`pinn_thermal.core.transfer`:
  1. ``resume_from`` (reanudación exacta de estado del optimizador/RNG/nubes).
  2. ``warm_start_from`` con ``transfer_mode="auto"|"weights"|"functional_readout"|"functional_pretrain"``.
  3. ``chain_from`` con ``chain_mode="residual"|"feature"`` (encadenamiento directo sobre un modelo previo).
  4. Inicialización por modelo discreto (`RBFFDInitializer` o `FEMInitializer`).
- Emisión uniforme de artefactos ``SolutionArtifact`` compatibles con Jupyter y Streamlit.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
import numbers
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Callable

import numpy as np
import torch

from pinn_thermal.core.artifact import (
    ARTIFACT_FORMAT,
    SolutionArtifact,
    build_specification,
    cpu_state,
)
from pinn_thermal.core.base_model import BasePINNModel, ModelRegistry
from pinn_thermal.core.device import DeviceManager
from pinn_thermal.core.evaluation import CRITERIA_STANDARD, evaluate_model
from pinn_thermal.core.manifest import atomic_replace, build_manifest, replace_json_atomic, write_json_atomic, write_manifest
from pinn_thermal.core.objective import DEFAULT_WEIGHTS, FOCUS_CRITERIA, PhysicsObjective
from pinn_thermal.core.problem import ThermalProblem2D, load_problem
from pinn_thermal.core.transfer import ChainedPINN, CompatibilityChecker, FunctionalTransfer, WeightTransfer

DEFAULT_POINTS: dict[str, int] = {"n_interior": 512, "n_layer": 256, "n_face": 128, "n_boundary": 128}


@dataclass
class TrainConfig:
    """Configuración declarativa y validable de un entrenamiento PINN."""

    adam_steps: int = 1500
    lbfgs_steps: int = 100
    lbfgs_iterations: int = 20
    lr: float = 2e-3
    width: int = 32
    depth: int = 3
    activation: str = "tanh"
    weights: dict[str, float] = field(default_factory=dict)
    points: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_POINTS))
    init_seed: int = 11
    sampling_seed: int = 101
    log_every: int = 100
    validation_every: int = 250
    device: str = "auto"
    dtype: str = "float32"
    sampling: str = "mixed"
    resample_every: int = 0
    attention: str | None = None
    input_normalization: str | None = None
    trace_parameterization: str = "standard"
    soil_envelope_gain: float = 1.0
    exact_energy: bool = False
    focused: bool = False
    focus_criterion: str = "residual"
    focus_surfaces: bool = True
    point_schedule_stages: int = 0
    point_schedule_start: float = 0.5
    full_audit: bool = True
    initializer: str | None = None  # None, 'rbf_readout', 'rbf_pretrain', 'fem_readout', 'fem_pretrain'
    initializer_settings: dict[str, Any] = field(default_factory=dict)
    # Rutas de continuación y encadenamiento
    resume_from: str | Path | None = None
    warm_start_from: str | Path | None = None
    transfer_mode: str = "auto"  # 'auto', 'weights', 'functional_readout', 'functional_readout_gls', 'functional_pretrain'
    transfer_steps: int = 150
    chain_from: str | Path | None = None
    chain_mode: str = "residual"  # 'residual', 'feature'
    freeze_chain_source: bool = True
    accuracy_stop: dict[str, Any] | None = None
    divergence_stop: dict[str, Any] | None = None
    study: str | None = None


class AccuracyStop:
    """Criterio opcional de parada temprana basado en métricas de evaluación."""

    DEFAULTS: dict[str, Any] = {
        "error_Tmax_abs_K": 0.01,
        "nrmse_pct": 0.1,
        "balance_pct": 0.5,
        "every": 100,
        "after_phase": "adam",
    }

    def __init__(self, settings: dict[str, Any], reference: np.ndarray | None = None, reference_label: str = "supplied") -> None:
        unknown = set(settings) - set(self.DEFAULTS)
        if unknown:
            raise ValueError(f"Claves desconocidas en accuracy_stop: {sorted(unknown)}.")
        self.settings = dict(self.DEFAULTS, **settings)
        self.reference = reference
        self.reference_label = reference_label
        self.checks: list[dict[str, Any]] = []
        self.triggered = False
        self.snapshot: dict[str, torch.Tensor] | None = None

    def check(self, row: dict[str, Any], model: BasePINNModel, force: bool = False) -> bool:
        if self.triggered:
            return True
        if self.settings["after_phase"] == "lbfgs" and row["phase"] != "lbfgs":
            return False
        step = int(row["step"])
        if self.checks and self.checks[-1]["step"] == step:
            return False
        if self.checks and not force and (step - self.checks[-1]["step"]) < int(self.settings["every"]):
            return False
        report, _ = evaluate_model(model, self.reference, self.reference_label, include_audit=True)
        vals = {
            "error_Tmax_abs_K": report.get("error_Tmax_abs_K"),
            "nrmse_pct": report.get("nrmse_pct"),
            "balance_pct": (report.get("balance") or {}).get("global_derived_pct"),
        }
        gates = {
            name: val is not None and np.isfinite(val) and val < float(self.settings[name])
            for name, val in vals.items()
        }
        self.triggered = bool(gates) and all(gates.values())
        self.checks.append({"step": step, "phase": row["phase"], "values": vals, "gates": gates, "met": self.triggered})
        if self.triggered:
            self.snapshot = cpu_state(model.state_dict())
        return self.triggered

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "settings": dict(self.settings),
            "triggered": self.triggered,
            "checks": self.checks,
            "reference_label": self.reference_label,
        }


class DivergenceGuard:
    """Detector temprano de divergencia y estancamiento para abortar corridas inestables."""

    DEFAULTS: dict[str, Any] = {
        "max_loss": 1e12,
        "explosion_factor": 1e4,
        "patience_explosion": 3,
        "growth_patience": 6,
        "min_growth_ratio": 1.05,
        "warmup_steps": 150,
        "lbfgs_explosion_factor": 100.0,
        "lbfgs_stagnation_checks": 8,
        "lbfgs_min_rel_improve": 1e-6,
        "check_nonfinite_grads": True,
    }

    def __init__(self, settings: dict[str, Any]) -> None:
        unknown = set(settings) - set(self.DEFAULTS)
        if unknown:
            raise ValueError(f"Claves desconocidas en divergence_stop: {sorted(unknown)}.")
        self.settings = dict(self.DEFAULTS, **settings)
        self.triggered: bool = False
        self.reason: str | None = None
        self.message: str | None = None
        self.step: int | None = None
        self.phase: str | None = None
        self.best_loss: float = float("inf")
        self.prev_loss: float | None = None
        self.explosion_streak: int = 0
        self.growth_streak: int = 0
        self.lbfgs_best_loss: float = float("inf")
        self.lbfgs_stagnation_streak: int = 0
        self.lbfgs_explosion_streak: int = 0
        self.lbfgs_early_stop: dict[str, Any] | None = None

    def _trip(self, reason: str, message: str, step: int, phase: str) -> bool:
        self.triggered = True
        self.reason = reason
        self.message = message
        self.step = int(step)
        self.phase = phase
        return True

    def notify_resample(self) -> None:
        """Reinicia la racha de crecimiento tras remuestreo para evitar falsos positivos."""
        self.prev_loss = None
        self.growth_streak = 0

    def check_step(self, step: int, phase: str, loss_val: float) -> bool:
        """Comprueba por paso si la pérdida es no finita o supera ``max_loss``."""
        if self.triggered:
            return True
        if not np.isfinite(loss_val):
            return self._trip(
                "non_finite_loss",
                f"Pérdida no finita ({loss_val}) en {phase} paso {step}.",
                step,
                phase,
            )
        max_loss = float(self.settings["max_loss"])
        if loss_val > max_loss:
            return self._trip(
                "loss_exceeded_max_loss",
                f"Pérdida ({loss_val:.3e}) superó max_loss ({max_loss:.3e}) en {phase} paso {step}.",
                step,
                phase,
            )
        return False

    def check_gradients(self, step: int, phase: str, model: BasePINNModel) -> bool:
        """Comprueba gradientes NaN/Inf tras ``backward()`` antes de actualizar pesos."""
        if self.triggered or not self.settings["check_nonfinite_grads"]:
            return self.triggered
        for p in model.parameters():
            if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
                return self._trip(
                    "non_finite_gradient",
                    f"Gradiente no finito detectado en {phase} paso {step}.",
                    step,
                    phase,
                )
        return False

    def check_adam_log(self, step: int, loss_val: float) -> bool:
        """Evalúa explosión relativa y crecimiento sostenido de pérdida en puntos de log."""
        if self.check_step(step, "adam", loss_val):
            return True
        if loss_val < self.best_loss:
            self.best_loss = loss_val

        if step >= int(self.settings["warmup_steps"]):
            baseline = max(self.best_loss, 1e-12)
            if loss_val > float(self.settings["explosion_factor"]) * baseline:
                self.explosion_streak += 1
            else:
                self.explosion_streak = 0
            if self.explosion_streak >= int(self.settings["patience_explosion"]):
                return self._trip(
                    "loss_explosion_vs_best",
                    f"Explosión de pérdida ({loss_val:.3e} > {self.settings['explosion_factor']}x mejor {baseline:.3e}) en adam paso {step}.",
                    step,
                    "adam",
                )

            if self.prev_loss is not None and loss_val > self.prev_loss * float(self.settings["min_growth_ratio"]):
                self.growth_streak += 1
            else:
                self.growth_streak = 0
            if self.growth_streak >= int(self.settings["growth_patience"]):
                return self._trip(
                    "sustained_loss_growth",
                    f"Crecimiento sostenido de pérdida durante {self.growth_streak} logs consecutivos hasta {loss_val:.3e} en adam paso {step}.",
                    step,
                    "adam",
                )

        self.prev_loss = loss_val
        return False

    def check_lbfgs_log(self, g_step: int, loss_val: float) -> str | None:
        """Evalúa divergencia dura o meseta/explosión en L-BFGS."""
        if self.check_step(g_step, "lbfgs", loss_val):
            return "diverged"

        baseline = min(self.best_loss, self.lbfgs_best_loss)
        if not np.isfinite(baseline):
            baseline = loss_val
        if loss_val > float(self.settings["lbfgs_explosion_factor"]) * max(baseline, 1e-12):
            self.lbfgs_explosion_streak += 1
            if self.lbfgs_explosion_streak >= 2:
                self.lbfgs_early_stop = {"reason": "lbfgs_loss_explosion", "step": int(g_step), "loss": float(loss_val)}
                return "plateau"
        else:
            self.lbfgs_explosion_streak = 0

        if loss_val < self.lbfgs_best_loss * (1.0 - float(self.settings["lbfgs_min_rel_improve"])):
            self.lbfgs_best_loss = loss_val
            self.lbfgs_stagnation_streak = 0
        else:
            self.lbfgs_stagnation_streak += 1
            if self.lbfgs_stagnation_streak >= int(self.settings["lbfgs_stagnation_checks"]):
                self.lbfgs_early_stop = {"reason": "lbfgs_plateau", "step": int(g_step), "loss": float(loss_val)}
                return "plateau"
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "settings": dict(self.settings),
            "triggered": self.triggered,
            "reason": self.reason,
            "message": self.message,
            "step": self.step,
            "phase": self.phase,
            "best_loss": self.best_loss if np.isfinite(self.best_loss) else None,
            "lbfgs_early_stop": self.lbfgs_early_stop,
        }


def _to_json_safe(value: Any) -> Any:
    """Convierte recursivamente objetos ``Path`` y estructuras anidadas a tipos JSON."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    return value


def _tree_is_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_tree_is_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_tree_is_finite(v) for v in value)
    if isinstance(value, np.ndarray):
        return (not np.issubdtype(value.dtype, np.number)) or bool(np.isfinite(value).all())
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return bool(np.isfinite(value))
    return True


def _save_collocation(path: Path, objective: PhysicsObjective) -> None:
    tmp = path.with_name("collocation.tmp.npz")
    arrays: dict[str, np.ndarray] = {f"interior_{rid}": pts.detach().cpu().numpy() for rid, pts in objective.interior.items()}
    arrays.update({f"interface_{idx}": entry[1].detach().cpu().numpy() for idx, entry in enumerate(objective.faces)})
    arrays.update({f"interface_right_{idx}": entry[3].detach().cpu().numpy() for idx, entry in enumerate(objective.faces)})
    arrays.update({f"boundary_{idx}": xy.detach().cpu().numpy() for idx, (_, xy, _) in enumerate(objective.pieces)})
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    atomic_replace(tmp, path)


class PINNTrainer:
    """Orquestador de entrenamiento, optimización híbrida, transferencia y persistencia."""

    def __init__(self, config: TrainConfig | None = None, criteria: dict[str, float] = CRITERIA_STANDARD) -> None:
        self.config = config or TrainConfig()
        self.criteria = dict(criteria)

    @staticmethod
    def _row(step: int, phase: str, total: torch.Tensor, parts: dict[str, torch.Tensor], lr: float) -> dict[str, Any]:
        return {
            "step": int(step),
            "phase": phase,
            "total": float(total.detach()),
            "lr": float(lr),
            **{k: float(v.detach()) for k, v in parts.items()},
        }

    def _fit_loop(
        self,
        model: BasePINNModel,
        objective: PhysicsObjective,
        device: torch.device,
        cfg: TrainConfig,
        on_log: Callable[[dict[str, Any], BasePINNModel], None] | None = None,
        on_checkpoint: Callable[[dict[str, Any], BasePINNModel, dict[str, Any]], None] | None = None,
        resume_state: dict[str, Any] | None = None,
        accuracy_stop: AccuracyStop | None = None,
        divergence_guard: DivergenceGuard | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        history: list[dict[str, Any]] = copy.deepcopy(resume_state["history"]) if resume_state else []
        best: dict[str, Any] = (
            copy.deepcopy(resume_state["best"])
            if resume_state
            else {"total": float("inf"), "state": None, "step": -1}
        )
        stopped = False
        diverged = False

        def keep(total: torch.Tensor, step: int) -> None:
            val = float(total.detach())
            if np.isfinite(val) and val < best["total"]:
                best.update(total=val, state=cpu_state(model.state_dict()), step=int(step))

        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=float(cfg.lr))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(int(cfg.adam_steps), 1), eta_min=float(cfg.lr) * 1e-2
        )

        def notify(row: dict[str, Any], opt: Any, sched: Any, phase: str, next_step: int) -> None:
            if accuracy_stop is not None:
                accuracy_stop.check(row, model, force=phase == "lbfgs")
            row["optimized_pde"] = float(objective.optimized_pde)
            best["training"] = {
                "model": cpu_state(model.state_dict()),
                "optimizer": cpu_state(opt.state_dict()),
                "scheduler": cpu_state(sched.state_dict()) if sched is not None else None,
                "phase": phase,
                "next_step": int(next_step),
                "completed_updates": int(row["step"]) + 1,
                "device_type": device.type,
                "adam_steps": int(cfg.adam_steps),
                "lbfgs_steps": int(cfg.lbfgs_steps),
                "best": cpu_state({"total": best["total"], "state": best["state"], "step": best["step"]}),
                "history": copy.deepcopy(history),
                "rng": cpu_state(DeviceManager.capture_rng(device)),
                "objective": cpu_state(objective.state_dict()),
            }
            if on_checkpoint is not None:
                on_checkpoint(row, model, best)
            if on_log is not None:
                on_log(row, model)

        adam_start = 0
        if resume_state is not None:
            model.load_state_dict(resume_state["model"])
            objective.load_state_dict(resume_state["objective"])
            DeviceManager.restore_rng(resume_state["rng"])
            if resume_state["phase"] == "adam":
                optimizer.load_state_dict(resume_state["optimizer"])
                if resume_state.get("scheduler"):
                    scheduler.load_state_dict(resume_state["scheduler"])
                adam_start = int(resume_state["next_step"])
            else:
                adam_start = int(cfg.adam_steps)
            best["training"] = cpu_state(resume_state)

        elapsed_phases = {"adam": 0.0, "lbfgs": 0.0}
        DeviceManager.synchronize(device)
        t0 = time.perf_counter()

        active_stage = None
        stages = int(cfg.point_schedule_stages)
        stage_interval = max((int(cfg.adam_steps) + stages - 1) // stages, 1) if stages else 1

        for step in range(adam_start, int(cfg.adam_steps)):
            if cfg.resample_every and step > 0 and step % int(cfg.resample_every) == 0:
                objective.resample()
                best.update(total=float("inf"), state=None, step=-1)
                if divergence_guard is not None:
                    divergence_guard.notify_resample()
            if stages:
                p_stage = min(step // stage_interval, stages - 1)
                if p_stage != active_stage:
                    frac = cfg.point_schedule_start + (1.0 - cfg.point_schedule_start) * p_stage / (stages - 1)
                    objective.set_point_fraction(frac)
                    active_stage = p_stage
            optimizer.zero_grad(set_to_none=True)
            total, parts, _ = objective()
            if divergence_guard is not None:
                total_val = float(total.detach())
                if divergence_guard.check_step(step, "adam", total_val):
                    if np.isfinite(total_val):
                        row = self._row(step, "adam", total, parts, optimizer.param_groups[0]["lr"])
                        history.append(row)
                        notify(row, optimizer, scheduler, "adam", step + 1)
                    diverged = True
                    break
            elif not torch.isfinite(total):
                raise FloatingPointError(f"La pérdida de Adam divergió en el paso {step}.")
            total.backward()
            if divergence_guard is not None and divergence_guard.check_gradients(step, "adam", model):
                diverged = True
                break
            optimizer.step()
            scheduler.step()
            objective.update_attention()

            if step % int(cfg.log_every) == 0 or step == int(cfg.adam_steps) - 1:
                total_e, parts_e, _ = objective()
                keep(total_e, step)
                row = self._row(step, "adam", total_e, parts_e, optimizer.param_groups[0]["lr"])
                history.append(row)
                notify(row, optimizer, scheduler, "adam", step + 1)
                if divergence_guard is not None and divergence_guard.check_adam_log(step, float(total_e.detach())):
                    diverged = True
                    break
                if accuracy_stop is not None and accuracy_stop.triggered:
                    best["stopped_state"] = cpu_state(model.state_dict())
                    best["stopped_step"] = step
                    stopped = True
                    break

        DeviceManager.synchronize(device)
        elapsed_phases["adam"] = time.perf_counter() - t0

        if int(cfg.lbfgs_steps) > 0 and not stopped and not diverged and cfg.attention is None:
            DeviceManager.synchronize(device)
            t1 = time.perf_counter()
            objective.set_point_fraction(1.0)
            continuing_lbfgs = resume_state is not None and resume_state["phase"] == "lbfgs"
            if best["state"] is not None and not continuing_lbfgs:
                model.load_state_dict(best["state"])
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            lbfgs = torch.optim.LBFGS(
                trainable_params,
                lr=1.0,
                max_iter=int(cfg.lbfgs_iterations),
                history_size=50,
                line_search_fn="strong_wolfe",
            )
            lbfgs_start = int(resume_state["next_step"]) if continuing_lbfgs and resume_state else 0

            def closure() -> torch.Tensor:
                lbfgs.zero_grad()
                tot, _, _ = objective()
                tot.backward()
                return tot

            for step in range(lbfgs_start, int(cfg.lbfgs_steps)):
                try:
                    lbfgs.step(closure)
                except (FloatingPointError, RuntimeError):
                    break
                total_e, parts_e, _ = objective()
                if not torch.isfinite(total_e):
                    if divergence_guard is not None:
                        divergence_guard.check_step(int(cfg.adam_steps) + step, "lbfgs", float(total_e.detach()))
                        diverged = True
                    break
                if step % max(int(cfg.log_every) // 20, 1) == 0 or step == int(cfg.lbfgs_steps) - 1:
                    g_step = int(cfg.adam_steps) + step
                    keep(total_e, g_step)
                    row = self._row(g_step, "lbfgs", total_e, parts_e, 1.0)
                    history.append(row)
                    notify(row, lbfgs, None, "lbfgs", step + 1)
                    if divergence_guard is not None:
                        action = divergence_guard.check_lbfgs_log(g_step, float(total_e.detach()))
                        if action == "diverged":
                            diverged = True
                            break
                        if action == "plateau":
                            break
                    if accuracy_stop is not None and accuracy_stop.triggered:
                        best["stopped_state"] = cpu_state(model.state_dict())
                        best["stopped_step"] = g_step
                        break
            DeviceManager.synchronize(device)
            elapsed_phases["lbfgs"] = time.perf_counter() - t1

        if "stopped_state" in best:
            model.load_state_dict(best["stopped_state"])
        elif best["state"] is not None:
            model.load_state_dict(best["state"])
        best["phase_elapsed_s"] = elapsed_phases
        best["diverged"] = diverged
        return history, best

    def train(
        self,
        problem_source: str | Path | dict[str, Any] | ThermalProblem2D,
        architecture: str,
        output_dir: str | Path,
        reference: np.ndarray | None = None,
        reference_label: str = "supplied",
        on_log: Callable[[dict[str, Any], BasePINNModel], None] | None = None,
        overwrite: bool = False,
    ) -> SolutionArtifact:
        """Ejecuta un entrenamiento completo y devuelve su ``SolutionArtifact`` persistido."""
        import pinn_thermal.models  # noqa: F401 - registra todas las arquitecturas

        cfg = copy.deepcopy(self.config)
        if cfg.attention and cfg.lbfgs_steps > 0:
            raise ValueError("L-BFGS requiere pesos de pérdida fijos; configure lbfgs_steps=0 al usar RBA/SA.")
        if cfg.focused and not cfg.resample_every:
            raise ValueError("El muestreo focalizado requiere resample_every > 0.")
        if cfg.focus_criterion not in FOCUS_CRITERIA:
            raise ValueError(f"focus_criterion debe ser uno de {FOCUS_CRITERIA}.")

        out = Path(output_dir)
        if out.exists() and any(out.iterdir()) and not overwrite:
            raise FileExistsError(f"El directorio de salida ya existe y no está vacío: {out}.")

        started = time.perf_counter()
        problem = load_problem(problem_source)
        device = DeviceManager.resolve(cfg.device)
        dtype = DeviceManager.resolve_dtype(device, cfg.dtype, allow_fallback=True)
        dtype_name = "float64" if dtype == torch.float64 else "float32"
        DeviceManager.seed_all(cfg.init_seed, device)

        out.mkdir(parents=True, exist_ok=True)
        val_seed = (int(cfg.sampling_seed) + 1_000_003) % (2 ** 32)
        seeds = {"init": int(cfg.init_seed), "sampling": int(cfg.sampling_seed), "validation": val_seed}

        # Construcción del modelo base
        base_model = ModelRegistry.build(
            problem=problem,
            architecture=architecture,
            width=int(cfg.width),
            depth=int(cfg.depth),
            activation=cfg.activation,
            input_normalization=cfg.input_normalization,
            trace_parameterization=cfg.trace_parameterization,
            soil_envelope_gain=float(cfg.soil_envelope_gain),
        ).to(device=device, dtype=dtype)

        transfer_report: dict[str, Any] | None = None
        resume_state: dict[str, Any] | None = None
        model: BasePINNModel = base_model
        chain_meta: dict[str, Any] | None = None

        # Ruta 1: Reanudación exacta
        if cfg.resume_from is not None:
            if cfg.warm_start_from is not None or cfg.chain_from is not None:
                raise ValueError("Use solo una ruta a la vez: resume_from, warm_start_from o chain_from.")
            src_art = SolutionArtifact.load(cfg.resume_from, device=device, selection="latest")
            if src_art.training_state is None or "optimizer" not in src_art.training_state:
                raise ValueError("El checkpoint indicado no contiene estado completo para reanudación exacta.")
            if src_art.problem.physics_sha256 != problem.physics_sha256:
                raise ValueError("La reanudación exacta exige exactamente la misma física (physics_sha256).")
            resume_state = src_art.training_state
            if src_art.model is not None:
                model = src_art.model.to(device=device, dtype=dtype)
            transfer_report = {"method": "exact_resume", "source": str(cfg.resume_from)}

        # Ruta 2: Transferencia de pesos o transferencia funcional automática/explícita
        elif cfg.warm_start_from is not None:
            if cfg.chain_from is not None:
                raise ValueError("No combine warm_start_from y chain_from en la misma ejecución.")
            src_art = SolutionArtifact.load(cfg.warm_start_from, device=device, selection="inference")
            mode = cfg.transfer_mode
            if mode == "auto":
                comp = CompatibilityChecker.check(src_art, model)
                if comp.compatible:
                    transfer_report = WeightTransfer.apply(src_art, model)
                else:
                    transfer_report = FunctionalTransfer.apply(
                        src_art, model, method="pretrain", steps=int(cfg.transfer_steps), lr=float(cfg.lr), seed=int(cfg.sampling_seed)
                    )
                    transfer_report["compatibility"] = comp.to_dict()
            elif mode == "weights":
                transfer_report = WeightTransfer.apply(src_art, model)
            elif mode.startswith("functional_"):
                sub_method = mode.replace("functional_", "", 1)
                transfer_report = FunctionalTransfer.apply(
                    src_art, model, method=sub_method, steps=int(cfg.transfer_steps), lr=float(cfg.lr), seed=int(cfg.sampling_seed)
                )
            else:
                raise ValueError(f"transfer_mode inválido: {mode!r}.")

        # Ruta 3: Encadenamiento directo (modelo previo como campo base residual o característica)
        elif cfg.chain_from is not None:
            src_art = SolutionArtifact.load(cfg.chain_from, device=device, selection="inference")
            model = ChainedPINN(
                source_artifact=src_art,
                target_model=base_model,
                chain_mode=cfg.chain_mode,
                freeze_source=cfg.freeze_chain_source,
            ).to(device=device, dtype=dtype)
            chain_meta = {
                "base_architecture": architecture,
                "chain_mode": cfg.chain_mode,
                "freeze_source": bool(cfg.freeze_chain_source),
                "source_specification": copy.deepcopy(src_art.specification),
            }
            transfer_report = {
                "method": "chain",
                "chain_mode": cfg.chain_mode,
                "source_id": src_art.id,
                "source_architecture": src_art.architecture,
            }

        # Pesos efectivos de pérdida
        eff_weights = dict(DEFAULT_WEIGHTS, **(cfg.weights or {}))
        if getattr(model, "exact_constraints", False):
            kept = ("pde", "interface_flux", "energy") if cfg.exact_energy else ("pde", "interface_flux")
            eff_weights = {k: (eff_weights[k] if k in kept else 0.0) for k in DEFAULT_WEIGHTS}

        points = dict(DEFAULT_POINTS, **(cfg.points or {}))
        config_record = _to_json_safe(asdict(cfg))
        config_record.update(
            {
                "architecture": model.architecture,
                "base_architecture": architecture,
                "dtype": dtype_name,
                "device": str(device),
                "weights": eff_weights,
                "points": points,
                "input_normalization": model.input_normalization,
            }
        )
        specification = build_specification(problem, config_record, seeds, model=model)
        if chain_meta is not None:
            specification["chain"] = chain_meta

        objective = PhysicsObjective(
            model=model,
            seed=int(cfg.sampling_seed),
            sampling=cfg.sampling,
            weights=eff_weights,
            attention=cfg.attention,
            exact_energy=cfg.exact_energy,
            focused=cfg.focused,
            focus_criterion=cfg.focus_criterion,
            focus_surfaces=cfg.focus_surfaces,
            **points,
        )
        val_objective = PhysicsObjective(model=model, seed=val_seed, weights=eff_weights, exact_energy=cfg.exact_energy, **points)

        # Ruta 4: Inicialización discreta RBF-FD o FEM previa al entrenamiento
        init_report: dict[str, Any] | None = None
        if cfg.initializer is not None and resume_state is None:
            if cfg.initializer.startswith("rbf_"):
                from pinn_thermal.models.rbf_init import RBFFDInitializer

                method_name = cfg.initializer.replace("rbf_", "", 1)
                init_report = RBFFDInitializer(**cfg.initializer_settings).apply(
                    model, objective, transfer=method_name, seed=int(cfg.sampling_seed)
                )
            elif cfg.initializer.startswith("fem_"):
                from pinn_thermal.models.rbf_init import FEMInitializer

                method_name = cfg.initializer.replace("fem_", "", 1)
                init_report = FEMInitializer(**cfg.initializer_settings).apply(
                    model, objective, output_dir=out / "fem_init", transfer=method_name
                )
            else:
                raise ValueError(f"Inicializador desconocido: {cfg.initializer!r}.")

        _save_collocation(out / "collocation.npz", objective)
        monitor = AccuracyStop(cfg.accuracy_stop, reference, reference_label) if cfg.accuracy_stop else None
        div_guard = DivergenceGuard(cfg.divergence_stop) if cfg.divergence_stop else None

        total_planned_steps = max(int(cfg.adam_steps) + (int(cfg.lbfgs_steps) if cfg.attention is None else 0), 1)
        replace_json_atomic(
            out / "progress.json",
            {
                "status": "running",
                "phase": "init",
                "step": 0,
                "completed_steps": 0,
                "adam_steps": int(cfg.adam_steps),
                "lbfgs_steps": int(cfg.lbfgs_steps) if cfg.attention is None else 0,
                "total_steps": total_planned_steps,
                "pct": 0.0,
                "loss_total": None,
                "elapsed_s": 0.0,
            },
        )

        best_val = {"loss": float("inf"), "step": None}
        last_val_step: int | None = None

        def on_ckpt(row: dict[str, Any], cur_model: BasePINNModel, best_state: dict[str, Any]) -> None:
            nonlocal last_val_step
            if last_val_step is None or int(row["step"]) - last_val_step >= int(cfg.validation_every):
                v_tot, _, _ = val_objective()
                v_loss = float(v_tot.detach())
                row["validation_loss"] = v_loss
                last_val_step = int(row["step"])
                if np.isfinite(v_loss) and v_loss < best_val["loss"]:
                    best_val.update(loss=v_loss, step=int(row["step"]))
            completed_steps = min(int(row["step"]) + 1, total_planned_steps)
            pct = 100.0 * completed_steps / total_planned_steps
            replace_json_atomic(
                out / "progress.json",
                {
                    "status": "running",
                    "phase": row["phase"],
                    "step": int(row["step"]),
                    "completed_steps": completed_steps,
                    "adam_steps": int(cfg.adam_steps),
                    "lbfgs_steps": int(cfg.lbfgs_steps) if cfg.attention is None else 0,
                    "total_steps": total_planned_steps,
                    "pct": round(pct, 2),
                    "loss_total": float(row["total"]),
                    "elapsed_s": round(time.perf_counter() - started, 2),
                    "best_validation": best_val,
                },
            )

        status = "completed"
        error_info: dict[str, Any] | None = None
        history: list[dict[str, Any]] = []
        best: dict[str, Any] = {}
        report: dict[str, Any] = {}
        arrays: dict[str, np.ndarray] = {}

        try:
            history, best = self._fit_loop(
                model=model,
                objective=objective,
                device=device,
                cfg=cfg,
                on_log=on_log,
                on_checkpoint=on_ckpt,
                resume_state=resume_state,
                accuracy_stop=monitor,
                divergence_guard=div_guard,
            )
            if best.get("diverged") and div_guard is not None and div_guard.triggered:
                status = "diverged"
                error_info = {
                    "type": "DivergenceDetected",
                    "reason": div_guard.reason,
                    "message": div_guard.message,
                    "step": div_guard.step,
                    "phase": div_guard.phase,
                }
                if best.get("state") is not None:
                    try:
                        cand_report, cand_arrays = evaluate_model(
                            model,
                            reference=reference,
                            reference_label=reference_label,
                            include_audit=cfg.full_audit,
                            criteria=self.criteria,
                        )
                        if _tree_is_finite(cand_report) and _tree_is_finite(cand_arrays):
                            report, arrays = cand_report, cand_arrays
                    except Exception:
                        report, arrays = {}, {}
            else:
                report, arrays = evaluate_model(
                    model, reference=reference, reference_label=reference_label, include_audit=cfg.full_audit, criteria=self.criteria
                )
                if not (_tree_is_finite(report) and _tree_is_finite(arrays)):
                    raise FloatingPointError("La evaluación final produjo métricas o temperaturas no finitas.")
        except Exception as exc:
            status = "failed"
            error_info = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
            write_json_atomic(out / "failure.json", {"error": error_info, "history": history, "config": config_record})

        total_loss = float(best["total"]) if best.get("state") is not None else float("nan")
        result: dict[str, Any] = {
            "format": ARTIFACT_FORMAT,
            "case": problem.id,
            "architecture": model.architecture,
            "model_kind": model.model_kind,
            "parameters": model.parameter_count(),
            "device": str(device),
            "dtype": dtype_name,
            "status": status,
            "config": config_record,
            "seeds": seeds,
            "physics_sha256": problem.physics_sha256,
            "fem_labels_used": 0,
            "reference_used_in_loss": False,
            "best_step": int(best.get("step", -1)),
            "best_loss": total_loss,
            "best_validation": best_val,
            "training_elapsed_s": best.get("phase_elapsed_s", {}),
            "elapsed_s": time.perf_counter() - started,
            "transfer_report": transfer_report,
            "initializer_report": init_report,
            "evaluation": report,
            "history": history,
            "error": error_info,
            "study": cfg.study,
        }
        if objective.attention is not None:
            result["attention_weights"] = objective.attention.summary()
        if monitor is not None:
            result["accuracy_stop"] = monitor.summary()
        if div_guard is not None:
            result["divergence_stop"] = div_guard.summary()

        artifact = SolutionArtifact(
            kind="pinn",
            problem=problem,
            specification=specification,
            result=result,
            model=model,
            training_state=best.get("training"),
            source_dir=out,
        )
        artifact.save(out, training_state=best.get("training"), evaluation_arrays=arrays or None)
        write_manifest(
            out,
            build_manifest(
                kind="pinn_training_run",
                command=sys.argv,
                seeds=seeds,
                extra={"case": problem.id, "architecture": model.architecture, "status": status, "physics_sha256": problem.physics_sha256},
            ),
            with_sources=False,
        )
        final_step = int(history[-1]["step"]) if history else -1
        if status == "diverged" and div_guard is not None and div_guard.step is not None:
            final_step = max(final_step, int(div_guard.step))
        final_completed = total_planned_steps if status == "completed" else min(max(final_step + 1, 0), total_planned_steps)
        final_prog = {
            "status": status,
            "phase": "done" if status == "completed" else status,
            "step": final_step,
            "completed_steps": final_completed,
            "adam_steps": int(cfg.adam_steps),
            "lbfgs_steps": int(cfg.lbfgs_steps) if cfg.attention is None else 0,
            "total_steps": total_planned_steps,
            "pct": 100.0 if status == "completed" else round(100.0 * final_completed / total_planned_steps, 2),
            "loss_total": total_loss if np.isfinite(total_loss) else None,
            "elapsed_s": round(time.perf_counter() - started, 2),
        }
        if status == "diverged" and div_guard is not None:
            final_prog["reason"] = div_guard.reason
        replace_json_atomic(out / "progress.json", final_prog)
        return artifact
