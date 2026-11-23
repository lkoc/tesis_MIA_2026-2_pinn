"""Modelos PINN enriquecidos con características de cúspide (``CUSP`` y ``CUSPR``).

Basados en el método CUSP (Tseng et al., 2023), donde las discontinuidades de pendiente
térmica provocadas por saltos de conductividad $k$ se representan agregando el valor
absoluto $|\\phi_\\Gamma(\\mathbf{x})|$ de funciones de distancia con signo hacia cada interfaz.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from pinn_thermal.core.base_model import BasePINNModel, ModelMetadata, ModelRegistry, build_mlp
from pinn_thermal.core.problem import RegionSpec, ThermalProblem2D


def _domain_length_scale(problem: ThermalProblem2D) -> float:
    x0, x1, y0, y1 = problem.bounds
    return max(math.hypot(x1 - x0, y1 - y0), 1e-12)


@ModelRegistry.register(
    "CUSP",
    ModelMetadata(
        name="CUSP",
        family="interface_enriched",
        partition="global",
        formulation="strong",
        authors="Tseng, Lin, Hu & Lai (CUSP-PINN)",
        year=2023,
        doi="10.1016/j.jcp.2023.112359",
        summary=(
            "Red neuronal global continua enriquecida con funciones cúspide |phi_Gamma(x)| "
            "para capturar quiebres en el gradiente térmico a través de interfaces sin partir la red."
        ),
        modifications=(
            "Genera automáticamente distancias con signo hacia círculos concéntricos, polígonos, "
            "subcajas y planos de estratificación declarados en el JSON del problema."
        ),
        supports_exact_constraints=False,
        supports_mixed_flux=False,
        tags=("interface-enriched", "global", "continuous-temperature"),
    ),
)
class CuspPINN(BasePINNModel):
    """``CUSP`` clásica: salida escalar continua en todo el dominio enriquecida con ``|phi|``."""

    model_kind = "cusp_global"

    def __init__(
        self,
        problem: ThermalProblem2D,
        architecture: str = "CUSP",
        width: int = 32,
        depth: int = 3,
        activation: str = "tanh",
        input_normalization: str | None = None,
        **_extra: object,
    ) -> None:
        super().__init__(
            problem=problem,
            architecture=architecture,
            width=width,
            depth=depth,
            activation=activation,
            partition="global",
            formulation="strong",
            input_normalization=input_normalization or "region_coordinates",
        )
        n_cusps = len(problem._level_set_specs())
        # Coordenadas normalizadas + características log-radiales + |level_sets|
        in_features = 2 + len(problem.source_disks) + n_cusps
        self._length_scale = _domain_length_scale(problem)
        self._amp = max(
            [problem.T_scale, *(problem.amplitude(r, "region") for r in problem.regions)]
        )
        self.networks = nn.ModuleList([build_mlp(in_features, self.width, self.depth, 1, self.activation)])
        self.offsets = nn.ParameterList([nn.Parameter(torch.zeros((), dtype=torch.float32))])

    def _features(self, xy: torch.Tensor) -> torch.Tensor:
        norm = self.problem.normalized_xy(xy)
        if self.input_normalization == "unit_box_v1":
            norm = (norm + 1.0) * 0.5
        log_feats = self.problem._source_distance_features(xy, self.problem.bounds)
        cusp = self.problem.level_sets(xy).abs() / self._length_scale
        return torch.cat([norm, *log_feats, cusp], dim=1)

    def field(self, region: RegionSpec, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        raw = self.networks[0](self._features(xy))[:, 0:1]
        temp = self.problem.T_ref + self.problem.T_scale * self.offsets[0] + self._amp * raw
        return temp, None

    def subnetwork_map(self) -> dict[str, nn.Module]:
        return {"global": self.networks[0]}


@ModelRegistry.register(
    "CUSPR",
    ModelMetadata(
        name="CUSPR",
        family="interface_enriched",
        partition="global",
        formulation="strong",
        authors="Tseng et al. (base CUSP) + adaptación multiescala de tesis",
        year=2025,
        doi="10.1016/j.jcp.2023.112359",
        summary=(
            "Variante escalada por región de CUSP: conserva la base enriquecida con cúspides |phi| "
            "sobre coordenadas globales, pero asigna sesgos y amplitudes físicas locales por región."
        ),
        modifications=(
            "Aporte del presente trabajo para resolver la disparidad de órdenes entre suelo "
            "y conductores de alta densidad de potencia sin sacrificar el acoplamiento global."
        ),
        supports_exact_constraints=False,
        supports_mixed_flux=False,
        tags=("interface-enriched", "global", "region-scaled", "thesis-retained"),
    ),
)
class CuspRegionScaledPINN(BasePINNModel):
    """``CUSPR``: red global enriquecida con ``|phi|`` y escalamiento físico ``Q*L^2/k`` por región."""

    model_kind = "cusp_region_scaled"

    def __init__(
        self,
        problem: ThermalProblem2D,
        architecture: str = "CUSPR",
        width: int = 32,
        depth: int = 3,
        activation: str = "tanh",
        input_normalization: str | None = None,
        **_extra: object,
    ) -> None:
        super().__init__(
            problem=problem,
            architecture=architecture,
            width=width,
            depth=depth,
            activation=activation,
            partition="global",
            formulation="strong",
            input_normalization=input_normalization or "region_coordinates",
        )
        n_cusps = len(problem._level_set_specs())
        in_features = problem.input_size(problem.regions[0], "global") + n_cusps
        self._length_scale = _domain_length_scale(problem)
        self.networks = nn.ModuleList(
            [build_mlp(in_features, self.width, self.depth, len(problem.regions), self.activation)]
        )
        self.offsets = nn.ParameterList(
            [nn.Parameter(torch.zeros((), dtype=torch.float32)) for _ in problem.regions]
        )

    def field(self, region: RegionSpec, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        base_feats = self.input_coordinates(region, xy)
        cusp = self.problem.level_sets(xy).abs() / self._length_scale
        raw = self.networks[0](torch.cat([base_feats, cusp], dim=1))[:, region.id:region.id + 1]
        amplitude = self.problem.amplitude(region, "region")
        temp = self.problem.T_ref + self.problem.T_scale * self.offsets[region.id] + amplitude * raw
        return temp, None

    def subnetwork_map(self) -> dict[str, nn.Module]:
        return {"global": self.networks[0]}
