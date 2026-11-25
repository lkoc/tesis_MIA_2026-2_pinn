"""Paquete ``pinn_thermal``: librería modular de PINN térmicas 2D y referencia FEniCSx.

Utiliza importación diferida en el nivel raíz para mantener compatibilidad total tanto
en entornos PyTorch (CPU/GPU/XPU/NPU) como en el subproceso FEniCSx libre de PyTorch.
"""

from __future__ import annotations

__version__ = "0.4.0"

_EXPORTS = {
    "AdaptiveRunTask": "pinn_thermal.core.parallel",
    "AmpacitySolver": "pinn_thermal.core.ampacity",
    "CaseCatalog": "pinn_thermal.core.problem",
    "ChainedPINN": "pinn_thermal.core.transfer",
    "CompatibilityChecker": "pinn_thermal.core.transfer",
    "DeviceManager": "pinn_thermal.core.device",
    "DivergenceGuard": "pinn_thermal.core.trainer",
    "FEniCSxAdapter": "pinn_thermal.fem.runner",
    "FunctionalTransfer": "pinn_thermal.core.transfer",
    "ModelRegistry": "pinn_thermal.core.base_model",
    "ParallelRunner": "pinn_thermal.core.parallel",
    "PINNTrainer": "pinn_thermal.core.trainer",
    "RunTask": "pinn_thermal.core.parallel",
    "SolutionArtifact": "pinn_thermal.core.artifact",
    "ThermalProblem2D": "pinn_thermal.core.problem",
    "TrainConfig": "pinn_thermal.core.trainer",
    "WeightTransfer": "pinn_thermal.core.transfer",
    "load_problem": "pinn_thermal.core.problem",
}


def __getattr__(name: str):
    if name in _EXPORTS:
        if name == "ModelRegistry":
            __import__("pinn_thermal.models")
        mod = __import__(_EXPORTS[name], fromlist=[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'pinn_thermal' has no attribute {name!r}")


__all__ = ["__version__", *sorted(_EXPORTS)]
