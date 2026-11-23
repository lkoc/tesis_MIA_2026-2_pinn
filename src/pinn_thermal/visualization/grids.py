"""Muestreo en grillas regulares, líneas de corte y evaluación desacoplada de artefactos.

Este módulo separa por completo la extracción de grillas 2D/1D para gráficos respecto de
las redes neuronales y del solucionador FEM: consume cualquier :class:`ThermalProblem2D`,
:class:`BasePINNModel` o :class:`SolutionArtifact`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pinn_thermal.core.problem import ThermalProblem2D


@dataclass(frozen=True)
class FieldGrid2D:
    """Resultado estructurado de muestreo sobre una malla cartesiana 2D."""

    x: np.ndarray
    y: np.ndarray
    T: np.ndarray
    region_ids: np.ndarray
    conductivity: np.ndarray
    source: np.ndarray
    qx: np.ndarray | None = None
    qy: np.ndarray | None = None
    T_reference: np.ndarray | None = None
    valid_mask: np.ndarray | None = None

    @property
    def error_map(self) -> np.ndarray | None:
        if self.T_reference is None:
            return None
        return self.T - self.T_reference


class GridSampler:
    """Generador desacoplado de grillas 2D, perfiles 1D y mapas de propiedades físicas."""

    @staticmethod
    def sample_problem_maps(
        problem: ThermalProblem2D,
        nx: int = 160,
        ny: int = 160,
        bounds: tuple[float, float, float, float] | None = None,
    ) -> dict[str, np.ndarray]:
        """Evalúa región propietaria, ``k(x, y)`` y ``Q(x, y)`` sin necesitar solución entrenada."""
        x0, x1, y0, y1 = bounds if bounds is not None else problem.bounds
        xs = np.linspace(float(x0), float(x1), int(nx))
        ys = np.linspace(float(y0), float(y1), int(ny))
        gx, gy = np.meshgrid(xs, ys)
        pts = np.c_[gx.ravel(), gy.ravel()]
        owners = problem.owner_index(pts)
        valid = owners >= 0

        k_arr = np.full(len(pts), np.nan, dtype=float)
        q_arr = np.full(len(pts), np.nan, dtype=float)
        if np.any(valid):
            k_arr[valid] = problem.field_numpy("conductivity", pts[valid, 0], pts[valid, 1])
            q_arr[valid] = problem.field_numpy("source", pts[valid, 0], pts[valid, 1])

        return {
            "x": xs,
            "y": ys,
            "gx": gx,
            "gy": gy,
            "region_ids": owners.reshape(int(ny), int(nx)),
            "conductivity": k_arr.reshape(int(ny), int(nx)),
            "source": q_arr.reshape(int(ny), int(nx)),
            "valid_mask": valid.reshape(int(ny), int(nx)),
        }

    @staticmethod
    def sample_artifact_grid(
        predictor: Any,
        problem: ThermalProblem2D | None = None,
        nx: int = 160,
        ny: int = 160,
        with_flux: bool = False,
        reference_predictor: Any | None = None,
        bounds: tuple[float, float, float, float] | None = None,
    ) -> FieldGrid2D:
        """Evalúa ``T`` (y flujo/referencia opcionales) desde cualquier artefacto o modelo."""
        prob = problem if problem is not None else predictor.problem
        maps = GridSampler.sample_problem_maps(prob, nx=nx, ny=ny, bounds=bounds)
        gx, gy = maps["gx"], maps["gy"]
        valid = maps["valid_mask"].ravel()
        pts = np.c_[gx.ravel(), gy.ravel()]

        t_grid = np.full(len(pts), np.nan, dtype=float)
        qx_grid = np.full(len(pts), np.nan, dtype=float) if with_flux else None
        qy_grid = np.full(len(pts), np.nan, dtype=float) if with_flux else None
        t_ref_grid = np.full(len(pts), np.nan, dtype=float)

        if np.any(valid):
            res = predictor.predict(pts[valid], with_flux=with_flux)
            t_grid[valid] = res["T"]
            if with_flux and "q" in res and qx_grid is not None and qy_grid is not None:
                qx_grid[valid] = res["q"][:, 0]
                qy_grid[valid] = res["q"][:, 1]

            if reference_predictor is not None:
                t_ref_grid[valid] = reference_predictor.predict(pts[valid], with_flux=False)["T"]
            else:
                exact = prob.reference(pts[valid])
                if exact is not None:
                    t_ref_grid[valid] = exact

        has_ref = np.any(np.isfinite(t_ref_grid))
        return FieldGrid2D(
            x=maps["x"],
            y=maps["y"],
            T=t_grid.reshape(int(ny), int(nx)),
            region_ids=maps["region_ids"],
            conductivity=maps["conductivity"],
            source=maps["source"],
            qx=qx_grid.reshape(int(ny), int(nx)) if qx_grid is not None else None,
            qy=qy_grid.reshape(int(ny), int(nx)) if qy_grid is not None else None,
            T_reference=t_ref_grid.reshape(int(ny), int(nx)) if has_ref else None,
            valid_mask=maps["valid_mask"],
        )

    @staticmethod
    def sample_line_cut(
        predictors: dict[str, Any],
        problem: ThermalProblem2D,
        start: tuple[float, float],
        end: tuple[float, float],
        n_points: int = 250,
    ) -> dict[str, Any]:
        """Muestrea uno o varios modelos/artefactos sobre un segmento rectilíneo ``start -> end``."""
        t = np.linspace(0.0, 1.0, int(n_points))
        xy = np.c_[
            start[0] + (end[0] - start[0]) * t,
            start[1] + (end[1] - start[1]) * t,
        ]
        dist = np.hypot(xy[:, 0] - start[0], xy[:, 1] - start[1])
        owners = problem.owner_index(xy)
        valid = owners >= 0

        series: dict[str, np.ndarray] = {}
        for label, pred in predictors.items():
            vals = np.full(len(xy), np.nan, dtype=float)
            if np.any(valid):
                vals[valid] = pred.predict(xy[valid], with_flux=False)["T"]
            series[label] = vals

        exact = np.full(len(xy), np.nan, dtype=float)
        if np.any(valid):
            ex_vals = problem.reference(xy[valid])
            if ex_vals is not None:
                exact[valid] = ex_vals
                series.setdefault("Exacta analítica", exact)

        k_line = np.full(len(xy), np.nan, dtype=float)
        if np.any(valid):
            k_line[valid] = problem.field_numpy("conductivity", xy[valid, 0], xy[valid, 1])

        return {
            "distance": dist,
            "x": xy[:, 0],
            "y": xy[:, 1],
            "region_ids": owners,
            "conductivity": k_line,
            "series": series,
        }
