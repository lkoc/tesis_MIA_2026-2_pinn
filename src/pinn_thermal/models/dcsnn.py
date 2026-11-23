"""Modelos globales capturadores de discontinuidades (``DCSNN``) y reescalados (``M0R``)."""

from __future__ import annotations

import math

import torch
from torch import nn

from pinn_thermal.core.base_model import BasePINNModel, ModelMetadata, ModelRegistry, build_mlp
from pinn_thermal.core.problem import RegionSpec, ThermalProblem2D


@ModelRegistry.register(
    "DCSNN",
    ModelMetadata(
        name="DCSNN",
        family="interface_enriched",
        partition="global",
        formulation="strong",
        authors="Hu, Lin & Lai (DCSNN)",
        year=2022,
        doi="10.1016/j.jcp.2022.111576",
        summary=(
            "Discontinuity-Capturing Neural Network: una sola red neuronal aumenta su vector "
            "de entrada con un indicador discontinuo por región z(x) y funciones de nivel."
        ),
        modifications=(
            "Se adapta el indicador por pieza a cualquier número arbitrario de regiones del JSON "
            "(cables multicapa o estratos del suelo) con amplitud térmica regional."
        ),
        supports_exact_constraints=False,
        supports_mixed_flux=False,
        tags=("interface-enriched", "global", "discontinuity-capturing"),
    ),
)
class DCSNNPINN(BasePINNModel):
    """Implementación 2D generalizada de ``DCSNN`` (Hu, Lin & Lai, 2022)."""

    model_kind = "dcsnn_global"

    def __init__(
        self,
        problem: ThermalProblem2D,
        architecture: str = "DCSNN",
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
        n_regions = len(problem.regions)
        n_levels = len(problem._level_set_specs())
        x0, x1, y0, y1 = problem.bounds
        self._length_scale = max(math.hypot(x1 - x0, y1 - y0), 1e-12)
        in_features = 2 + n_regions + n_levels
        self.networks = nn.ModuleList([build_mlp(in_features, self.width, self.depth, 1, self.activation)])
        self.offsets = nn.ParameterList([nn.Parameter(torch.zeros((), dtype=torch.float32))])

    def field(self, region: RegionSpec, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        norm = self.problem.normalized_xy(xy)
        if self.input_normalization == "unit_box_v1":
            norm = (norm + 1.0) * 0.5
        indicator = xy.new_zeros((len(xy), len(self.problem.regions)))
        indicator[:, region.id] = 1.0
        levels = self.problem.level_sets(xy) / self._length_scale
        raw = self.networks[0](torch.cat([norm, indicator, levels], dim=1))[:, 0:1]
        amplitude = self.problem.amplitude(region, "region")
        temp = self.problem.T_ref + self.problem.T_scale * self.offsets[0] + amplitude * raw
        return temp, None

    def subnetwork_map(self) -> dict[str, nn.Module]:
        return {"global": self.networks[0]}


@ModelRegistry.register(
    "M0R",
    ModelMetadata(
        name="M0R",
        family="baseline_scaled",
        partition="global",
        formulation="strong",
        authors="Raissi et al. (base) + adaptación escalada de tesis",
        year=2025,
        doi="10.1016/j.jcp.2018.10.045",
        summary=(
            "PINN global escalada por región (M0R): mantiene una red compartida única pero "
            "escala cada cabeza de salida con su temperatura característica física Q*L^2/k."
        ),
        modifications=(
            "Corrige la subestimación de picos térmicos observada en M0 cuando coexisten fuentes "
            "intensas en conductores pequeños dentro de dominios extensos de suelo."
        ),
        supports_exact_constraints=False,
        supports_mixed_flux=False,
        tags=("global", "region-scaled", "thesis-retained"),
    ),
)
class RegionScaledGlobalPINN(BasePINNModel):
    """PINN global ``M0R`` con escalamiento físico local por subregión."""

    model_kind = "global_region_scaled"

    def __init__(
        self,
        problem: ThermalProblem2D,
        architecture: str = "M0R",
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
        in_features = problem.input_size(problem.regions[0], "global")
        self.networks = nn.ModuleList(
            [build_mlp(in_features, self.width, self.depth, len(problem.regions), self.activation)]
        )
        self.offsets = nn.ParameterList(
            [nn.Parameter(torch.zeros((), dtype=torch.float32)) for _ in problem.regions]
        )

    def field(self, region: RegionSpec, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        features = self.input_coordinates(region, xy)
        raw = self.networks[0](features)[:, region.id:region.id + 1]
        amplitude = self.problem.amplitude(region, "region")
        temp = self.problem.T_ref + self.problem.T_scale * self.offsets[region.id] + amplitude * raw
        return temp, None

    def subnetwork_map(self) -> dict[str, nn.Module]:
        return {"global": self.networks[0]}
