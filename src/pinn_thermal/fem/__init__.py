"""Subsistema de referencia por elementos finitos (Gmsh + DolfinX + PETSc P2).

Importa :mod:`pinn_thermal.fem.runner` de forma diferida (PEP 562) para garantizar que
el trabajador :mod:`pinn_thermal.fem.worker` pueda ejecutarse en entornos sin PyTorch.
"""

from __future__ import annotations

from pinn_thermal.fem.mesh_resolution import MeshSizingPlan, build_mesh_plan

__all__ = [
    "MeshSizingPlan",
    "build_mesh_plan",
    "FEniCSxAdapter",
    "windows_to_wsl_path",
]


def __getattr__(name: str):
    if name in ("FEniCSxAdapter", "windows_to_wsl_path"):
        from pinn_thermal.fem import runner

        return getattr(runner, name)
    raise AttributeError(f"El módulo 'pinn_thermal.fem' no expone {name!r}.")
