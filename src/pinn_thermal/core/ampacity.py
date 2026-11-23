"""Cálculo acoplado electrotérmico y búsqueda acotada de ampacidad (corriente límite).

Implementa la formulación descrita en la tesis para conductores con resistencia DC
dependiente de la temperatura:
    R(T) = R_20 * (1 + alpha_20 * (T_media - 20))

1. **Acoplamiento de punto fijo relajado** para una corriente fija ``I``:
   Actualiza la temperatura media del conductor con relajación ``0.5 * T_prev + 0.5 * T_new``
   hasta satisfacer tolerancia térmica (``< 0.001 K``) y relativa de potencia (``< 1e-4``).
2. **Encerramiento y bisección acotada** de la corriente máxima ``I_max`` tal que
   ``T_max(I_max) = T_lim`` dentro de ``|T_max - T_lim| <= 0.05 K`` y ``(I_U - I_L) / I_U <= 1e-3``.

El solucionador acepta cualquier función evaluadora ``field_solver(power_W_m)``, o bien
las fachadas de alto nivel :meth:`AmpacitySolver.solve_with_pinn` y
:meth:`AmpacitySolver.solve_with_fem`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np


@dataclass
class AmpacityResult:
    """Resultado completo y trazable de una búsqueda acotada de ampacidad."""

    converged: bool
    I_lower_A: float
    I_upper_A: float
    I_mid_A: float
    Tmax_at_mid_C: float
    T_limit_C: float
    bisection_steps: int
    total_solves: int
    reason: str
    trace: list[dict[str, float]]
    power_W_m: float = 0.0

    @property
    def peak_temperature_C(self) -> float:
        return self.Tmax_at_mid_C

    @property
    def history(self) -> list[dict[str, float]]:
        return self.trace


class AmpacitySolver:
    """Solucionador desacoplado de punto fijo electrotérmico y bisección de ampacidad."""

    def __init__(
        self,
        R20_ohm_m: float = 1.0e-4,
        alpha20_1_K: float = 0.00393,
        T_limit_C: float = 90.0,
        T_ref_C: float = 20.0,
        relaxation: float = 0.5,
        coupling_tol_K: float = 1e-3,
        source_tol_rel: float = 1e-4,
        max_coupling_iters: int = 100,
        max_bracket_doublings: int = 12,
        max_bisection_steps: int = 60,
        bisection_rel_tol: float = 1e-3,
        bisection_temp_tol_K: float = 0.05,
        target_temp_C: float | None = None,
        max_iters: int | None = None,
    ) -> None:
        self.R20 = float(R20_ohm_m)
        self.alpha20 = float(alpha20_1_K)
        self.T_limit = float(target_temp_C) if target_temp_C is not None else float(T_limit_C)
        self.T_ref = float(T_ref_C)
        self.relaxation = float(relaxation)
        self.coupling_tol_K = float(coupling_tol_K)
        self.source_tol_rel = float(source_tol_rel)
        self.max_coupling_iters = int(max_coupling_iters)
        self.max_bracket_doublings = int(max_bracket_doublings)
        self.max_bisection_steps = int(max_iters) if max_iters is not None else int(max_bisection_steps)
        self.bisection_rel_tol = float(bisection_rel_tol)
        self.bisection_temp_tol_K = float(bisection_temp_tol_K)

    def effective_power_W_m(self, current_A: float, T_mean_C: float) -> float:
        """Calcula las pérdidas Joule lineales [W/m] a temperatura media del conductor."""
        resistance = self.R20 * (1.0 + self.alpha20 * (T_mean_C - self.T_ref))
        if resistance <= 0.0:
            raise ValueError(f"La resistencia efectiva se volvió no positiva ({resistance:.4g} ohm/m).")
        return (float(current_A) ** 2) * resistance

    def solve_coupled_temperature(
        self,
        current_A: float,
        field_solver: Callable[[float], tuple[float, float]],
        initial_T_C: float | None = None,
    ) -> dict[str, Any]:
        """Ejecuta el punto fijo relajado para una corriente fija ``current_A``.

        ``field_solver(power_W_m)`` debe devolver ``(T_max_conductor_C, T_mean_conductor_C)``.
        """
        T_bar = float(self.T_ref if initial_T_C is None else initial_T_C)
        q_prev = self.effective_power_W_m(current_A, T_bar)
        q_scale = max(self.effective_power_W_m(max(current_A, 1.0), self.T_ref), 1e-9)

        t_max = T_bar
        for it in range(1, self.max_coupling_iters + 1):
            t_max, t_mean_new = field_solver(q_prev)
            t_next = (1.0 - self.relaxation) * T_bar + self.relaxation * float(t_mean_new)
            q_next = self.effective_power_W_m(current_A, t_next)
            dT = abs(t_next - T_bar)
            dq_rel = abs(q_next - q_prev) / max(abs(q_next), q_scale)
            T_bar, q_prev = t_next, q_next
            if dT < self.coupling_tol_K and dq_rel < self.source_tol_rel:
                return {
                    "converged": True,
                    "iterations": it,
                    "Tmax_C": float(t_max),
                    "Tmean_C": float(T_bar),
                    "power_W_m": float(q_next),
                }
        return {
            "converged": False,
            "iterations": self.max_coupling_iters,
            "Tmax_C": float(t_max),
            "Tmean_C": float(T_bar),
            "power_W_m": float(q_prev),
        }

    def find_ampacity(
        self,
        initial_upper_A: float,
        field_solver: Callable[[float], tuple[float, float]],
    ) -> AmpacityResult:
        """Encierra el intervalo ``[I_L, I_U]`` y aplica bisección acotada hasta tolerancia."""
        if initial_upper_A <= 0.0:
            raise ValueError("initial_upper_A debe ser positivo.")

        trace: list[dict[str, float]] = []
        solves = 0

        def eval_current(curr: float) -> float:
            nonlocal solves
            solves += 1
            res = self.solve_coupled_temperature(curr, field_solver)
            if not res["converged"]:
                raise RuntimeError(f"El acoplamiento electrotérmico no convergió para I={curr:.2f} A.")
            trace.append(
                {
                    "current_A": float(curr),
                    "Tmax_C": res["Tmax_C"],
                    "Tmean_C": res["Tmean_C"],
                    "power_W_m": res["power_W_m"],
                }
            )
            return float(res["Tmax_C"])

        I_L = 0.0
        T_L = eval_current(I_L)
        if T_L > self.T_limit:
            p0 = trace[-1]["power_W_m"] if trace else 0.0
            return AmpacityResult(False, I_L, I_L, I_L, T_L, self.T_limit, 0, solves, "T(0 A) excede el límite térmico.", trace, p0)

        I_U = float(initial_upper_A)
        T_U = eval_current(I_U)
        for _ in range(self.max_bracket_doublings):
            if T_U > self.T_limit:
                break
            I_L, T_L = I_U, T_U
            I_U *= 2.0
            T_U = eval_current(I_U)
        else:
            pu = trace[-1]["power_W_m"] if trace else 0.0
            return AmpacityResult(
                False, I_L, I_U, I_U, T_U, self.T_limit, 0, solves,
                "No se logró acotar T_max > T_lim tras máximo de duplicaciones.", trace, pu,
            )

        I_M = 0.5 * (I_L + I_U)
        T_M = T_L
        for step in range(1, self.max_bisection_steps + 1):
            I_M = 0.5 * (I_L + I_U)
            T_M = eval_current(I_M)
            if T_M > self.T_limit:
                I_U = I_M
            else:
                I_L = I_M
            if abs(T_M - self.T_limit) <= self.bisection_temp_tol_K and (I_U - I_L) / max(I_U, 1e-12) <= self.bisection_rel_tol:
                pm = trace[-1]["power_W_m"]
                return AmpacityResult(True, I_L, I_U, I_M, T_M, self.T_limit, step, solves, "converged", trace, pm)

        pm = trace[-1]["power_W_m"] if trace else 0.0
        return AmpacityResult(
            False, I_L, I_U, I_M, T_M, self.T_limit, self.max_bisection_steps, solves,
            "Se alcanzó el máximo de bisecciones sin cumplir ambas tolerancias.", trace, pm,
        )

    @staticmethod
    def _conductor_probe_points(problem: Any, per_region: int = 80) -> np.ndarray:
        cond_regs = [r for r in problem.regions if r.shape.shape_type == "disk"]
        if not cond_regs:
            cond_regs = [r for r in problem.regions if r.kind == "conductor"] or [problem.regions[0]]
        blocks = [problem.sample_region_interior(r, per_region, rng=17 + r.id) for r in cond_regs]
        return np.vstack(blocks)

    def solve_with_pinn(
        self,
        problem: Any,
        architecture: str = "CUSPR",
        config: Any | None = None,
        work_dir: str | Path = "ampacity_pinn_runs",
        initial_upper_A: float = 500.0,
    ) -> AmpacityResult:
        """Calcula ampacidad acoplada usando ``PINNTrainer`` como evaluador térmico 2D."""
        from pinn_thermal.core.problem import ThermalProblem2D, load_problem
        from pinn_thermal.core.trainer import PINNTrainer, TrainConfig

        base_prob = load_problem(problem)
        probe_xy = self._conductor_probe_points(base_prob)
        base_cfg = copy.deepcopy(config) if config is not None else TrainConfig(adam_steps=300, lbfgs_steps=20, width=24, depth=3)
        base_cfg.resume_from = None
        base_cfg.chain_from = None
        root = Path(work_dir)
        root.mkdir(parents=True, exist_ok=True)
        prev_dir: Path | None = None
        counter = 0

        def _solve_for_power(power_W_m: float) -> tuple[float, float]:
            nonlocal prev_dir, counter
            counter += 1
            mod_doc = with_conductor_power(base_prob.doc, power_W_m)
            mod_doc["id"] = f"{base_prob.id}_amp_{counter}"
            mod_prob = ThermalProblem2D.from_json(mod_doc)
            run_cfg = copy.deepcopy(base_cfg)
            if prev_dir is not None:
                run_cfg.warm_start_from = str(prev_dir)
                run_cfg.transfer_mode = "auto"
            out_dir = root / f"eval_{counter:03d}"
            art = PINNTrainer(run_cfg).train(mod_prob, architecture, out_dir, overwrite=True)
            prev_dir = out_dir
            t_vals = art.predict(probe_xy)["T"]
            return float(np.max(t_vals)), float(np.mean(t_vals))

        return self.find_ampacity(initial_upper_A=initial_upper_A, field_solver=_solve_for_power)

    def solve_with_fem(
        self,
        problem: Any,
        work_dir: str | Path = "ampacity_fem_runs",
        level: int = 1,
        initial_upper_A: float = 500.0,
    ) -> AmpacityResult:
        """Calcula ampacidad acoplada usando FEniCSx (WSL2/nativo/Docker) como evaluador."""
        from pinn_thermal.core.problem import ThermalProblem2D, load_problem
        from pinn_thermal.fem.runner import FEniCSxAdapter

        base_prob = load_problem(problem)
        probe_xy = self._conductor_probe_points(base_prob)
        adapter = FEniCSxAdapter()
        root = Path(work_dir)
        root.mkdir(parents=True, exist_ok=True)
        counter = 0

        def _solve_for_power(power_W_m: float) -> tuple[float, float]:
            nonlocal counter
            counter += 1
            mod_doc = with_conductor_power(base_prob.doc, power_W_m)
            mod_doc["id"] = f"{base_prob.id}_fem_amp_{counter}"
            mod_prob = ThermalProblem2D.from_json(mod_doc)
            out_dir = root / f"fem_{counter:03d}"
            art = adapter.solve(mod_prob, output_dir=out_dir, level=level, reuse_if_valid=False)
            t_vals = art.predict(probe_xy)["T"]
            return float(np.max(t_vals)), float(np.mean(t_vals))

        return self.find_ampacity(initial_upper_A=initial_upper_A, field_solver=_solve_for_power)


def with_conductor_power(problem_doc: dict[str, Any], power_W_m: float) -> dict[str, Any]:
    """Genera una copia del JSON sustituyendo la fuente volumétrica por ``power_W_m / area`` en conductores."""
    doc = copy.deepcopy(problem_doc)
    for region in doc["regions"]:
        shape = region["shape"]
        is_cond = region.get("kind") == "conductor" or (
            shape["type"] == "disk" and isinstance(region.get("source", 0.0), (int, float)) and float(region.get("source", 0.0) or 0.0) != 0.0
        )
        if is_cond and shape["type"] == "disk":
            area = math.pi * float(shape["radius_m"]) ** 2
            region["source"] = float(power_W_m) / area
    return doc
