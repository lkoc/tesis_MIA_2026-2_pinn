"""Política de refinamiento geométrico y tamaños característicos de malla para Gmsh + FEniCSx."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from pinn_thermal.core.problem import RegionSpec, ThermalProblem2D


@dataclass(frozen=True)
class MeshSizingPlan:
    """Plan numérico de tamaños de elemento ``lc`` por región para un nivel FEM dado."""

    level: int
    domain_lc: float
    min_lc: float
    region_lc: dict[int, float]


def build_mesh_plan(problem: ThermalProblem2D, level: int = 2) -> MeshSizingPlan:
    """Calcula tamaños característicos adaptados al espesor mínimo de capas y nivel (1..4)."""
    lvl = max(1, int(level))
    refine = 1.65 ** (lvl - 1)
    x0, x1, y0, y1 = problem.bounds
    span = max(x1 - x0, y1 - y0, 1e-6)
    base_domain_lc = span / (28.0 * refine)

    region_lc: dict[int, float] = {}
    min_lc = base_domain_lc

    for region in problem.regions:
        s = region.shape
        if s.shape_type == "disk":
            r = float(s.radius_m)  # type: ignore[attr-defined]
            lc = max(r / (5.0 * refine), span * 2e-5)
        elif s.shape_type == "annulus":
            ri = float(s.inner_radius_m)  # type: ignore[attr-defined]
            ro = float(s.outer_radius_m)  # type: ignore[attr-defined]
            thickness = max(ro - ri, 1e-6)
            lc = max(min(thickness / (3.0 * refine), ro / (6.0 * refine), ri / (6.0 * refine)), span * 2e-5)
        elif s.shape_type == "rectangle":
            bx0, bx1, by0, by1 = s.bounds  # type: ignore[attr-defined]
            sub_span = min(bx1 - bx0, by1 - by0)
            lc = max(min(base_domain_lc, sub_span / (10.0 * refine)), span * 2e-5)
        else:
            verts = s.vertices  # type: ignore[attr-defined]
            xs = [v[0] for v in verts]
            ys = [v[1] for v in verts]
            sub_span = max(min(max(xs) - min(xs), max(ys) - min(ys)), 1e-6)
            lc = max(min(base_domain_lc, sub_span / (8.0 * refine)), span * 2e-5)
        region_lc[region.id] = float(lc)
        min_lc = min(min_lc, float(lc))

    return MeshSizingPlan(
        level=lvl,
        domain_lc=float(base_domain_lc),
        min_lc=float(min_lc),
        region_lc=region_lc,
    )
