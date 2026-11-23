"""Registro automático de todas las arquitecturas PINN y métodos de inicialización."""

from pinn_thermal.models import baseline as _baseline  # noqa: F401
from pinn_thermal.models import cusp as _cusp  # noqa: F401
from pinn_thermal.models import dcsnn as _dcsnn  # noqa: F401
from pinn_thermal.models import deepxde_adapter as _deepxde_adapter  # noqa: F401
from pinn_thermal.models import exact as _exact  # noqa: F401
from pinn_thermal.models import fbpinn as _fbpinn  # noqa: F401

from pinn_thermal.models.baseline import M0GlobalPINN, M1SubdomainPINN, M2MixedFluxPINN
from pinn_thermal.models.cusp import CuspPINN, CuspRegionScaledPINN
from pinn_thermal.models.dcsnn import DCSNNPINN, RegionScaledGlobalPINN
from pinn_thermal.models.deepxde_adapter import DeepXDEModelAdapter
from pinn_thermal.models.exact import ExactBoundaryPINN, ExactCorrectorPINN, ExactInterfaceCablePINN
from pinn_thermal.models.fbpinn import FBPINNPINN
from pinn_thermal.models.rbf_init import FEMInitializer, RBFFDInitializer, RBFFDSolver

__all__ = [
    "M0GlobalPINN",
    "M1SubdomainPINN",
    "M2MixedFluxPINN",
    "CuspPINN",
    "CuspRegionScaledPINN",
    "DCSNNPINN",
    "RegionScaledGlobalPINN",
    "FBPINNPINN",
    "ExactInterfaceCablePINN",
    "ExactBoundaryPINN",
    "ExactCorrectorPINN",
    "DeepXDEModelAdapter",
    "RBFFDSolver",
    "RBFFDInitializer",
    "FEMInitializer",
]
