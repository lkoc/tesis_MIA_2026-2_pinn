"""Núcleo modular (Capa 1) de ``pinn_thermal``.

Mantiene importación diferida (``__getattr__``) para los submódulos basados en PyTorch,
permitiendo que el núcleo geométrico/esquema y el worker de FEniCSx importen
``ThermalProblem2D``, ``FieldSpec`` o ``ProblemValidator`` sin requerir ``torch``.
"""

from __future__ import annotations

from pinn_thermal.core.fields import FieldSpec, evaluate_field_numpy
from pinn_thermal.core.problem import CaseCatalog, ThermalProblem2D, compute_physics_sha256, load_problem
from pinn_thermal.core.schema import ProblemValidator, convert_legacy_case, validate_problem
from pinn_thermal.core.shapes import Shape2D

_LAZY_TORCH_EXPORTS = {
    "AcceptanceGate": "pinn_thermal.core.evaluation",
    "AdaptiveRunTask": "pinn_thermal.core.parallel",
    "AmpacitySolver": "pinn_thermal.core.ampacity",
    "BasePINNModel": "pinn_thermal.core.base_model",
    "ChainedPINN": "pinn_thermal.core.transfer",
    "CompatibilityChecker": "pinn_thermal.core.transfer",
    "DeviceManager": "pinn_thermal.core.device",
    "DivergenceGuard": "pinn_thermal.core.trainer",
    "ErrorMetrics": "pinn_thermal.core.evaluation",
    "FunctionalTransfer": "pinn_thermal.core.transfer",
    "ModelMetadata": "pinn_thermal.core.base_model",
    "ModelRegistry": "pinn_thermal.core.base_model",
    "ParallelRunner": "pinn_thermal.core.parallel",
    "PhysicsAuditor": "pinn_thermal.core.evaluation",
    "PhysicsObjective": "pinn_thermal.core.objective",
    "PINNTrainer": "pinn_thermal.core.trainer",
    "RunTask": "pinn_thermal.core.parallel",
    "SolutionArtifact": "pinn_thermal.core.artifact",
    "TrainConfig": "pinn_thermal.core.trainer",
    "WeightTransfer": "pinn_thermal.core.transfer",
}


def __getattr__(name: str):
    if name in _LAZY_TORCH_EXPORTS:
        mod = __import__(_LAZY_TORCH_EXPORTS[name], fromlist=[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'pinn_thermal.core' has no attribute {name!r}")


__all__ = [
    "CaseCatalog",
    "FieldSpec",
    "ProblemValidator",
    "Shape2D",
    "ThermalProblem2D",
    "compute_physics_sha256",
    "convert_legacy_case",
    "evaluate_field_numpy",
    "load_problem",
    "validate_problem",
    *sorted(_LAZY_TORCH_EXPORTS),
]
