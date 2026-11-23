"""Modelos PINN base: ``M0`` (global), ``M1`` (descomposición por subdominios) y ``M2`` (mixto flujo-temperatura).

Estos tres modelos constituyen la línea base clásica sobre la que se comparan las
formulaciones enriquecidas (``CUSP``, ``CUSPR``, ``DCSNN``, ``FBPINN`` y variantes exactas).
"""

from __future__ import annotations

import torch
from torch import nn

from pinn_thermal.core.base_model import BasePINNModel, ModelMetadata, ModelRegistry, build_mlp
from pinn_thermal.core.problem import RegionSpec, ThermalProblem2D


class _OffsetMLPModel(BasePINNModel):
    """Base interna compartida para redes MLP con sesgos regionales entrenables (``offsets``)."""

    def __init__(
        self,
        problem: ThermalProblem2D,
        architecture: str,
        width: int = 32,
        depth: int = 3,
        activation: str = "tanh",
        partition: str = "region",
        formulation: str = "strong",
        input_normalization: str | None = None,
    ) -> None:
        super().__init__(
            problem=problem,
            architecture=architecture,
            width=width,
            depth=depth,
            activation=activation,
            partition=partition,
            formulation=formulation,
            input_normalization=input_normalization or "region_coordinates",
        )
        self.networks = nn.ModuleList()
        self.offsets = nn.ParameterList()

    def _raw_with_offset(
        self,
        network: nn.Module,
        offset: nn.Parameter,
        features: torch.Tensor,
        amplitude: float,
        channel: int = 0,
    ) -> torch.Tensor:
        raw = network(features)[:, channel:channel + 1]
        return self.problem.T_ref + self.problem.T_scale * offset + float(amplitude) * raw


@ModelRegistry.register(
    "M0",
    ModelMetadata(
        name="M0",
        family="baseline",
        partition="global",
        formulation="strong",
        authors="Raissi, Perdikaris & Karniadakis",
        year=2019,
        doi="10.1016/j.jcp.2018.10.045",
        summary=(
            "PINN global clásica en formulación fuerte de segundo orden: una única red "
            "compartida recibe coordenadas normalizadas y salidas por región con escala global."
        ),
        modifications=(
            "Se adaptó la entrada global agregando codificación geométrica por región "
            "(coordenadas locales rellenadas, selector one-hot y distancias logarítmicas a fuentes) "
            "junto con sesgos regionales entrenables para estabilizar la escala térmica."
        ),
        supports_exact_constraints=False,
        supports_mixed_flux=False,
        tags=("baseline", "global", "strong-form"),
    ),
)
class M0GlobalPINN(_OffsetMLPModel):
    """PINN global única ``M0`` con escala térmica uniforme ``T_scale`` por defecto."""

    model_kind = "baseline_global"

    def __init__(
        self,
        problem: ThermalProblem2D,
        architecture: str = "M0",
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
            input_normalization=input_normalization,
        )
        in_features = problem.input_size(problem.regions[0], "global")
        n_regions = len(problem.regions)
        self.networks.append(build_mlp(in_features, self.width, self.depth, n_regions, self.activation))
        for _ in problem.regions:
            self.offsets.append(nn.Parameter(torch.zeros((), dtype=torch.float32)))

    def field(self, region: RegionSpec, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        features = self.input_coordinates(region, xy)
        raw = self.networks[0](features)[:, region.id:region.id + 1]
        temp = self.problem.T_ref + self.problem.T_scale * (self.offsets[region.id] + raw)
        return temp, None


@ModelRegistry.register(
    "M1",
    ModelMetadata(
        name="M1",
        family="baseline",
        partition="region",
        formulation="strong",
        authors="Jagtap, Kharazmi & Karniadakis (cPINN / XPINN)",
        year=2020,
        doi="10.1016/j.cma.2020.113028",
        summary=(
            "Descomposición conservativa de dominio (cPINN/XPINN) en formulación fuerte: "
            "asigna una subred neuronal independiente a cada material o subregión física."
        ),
        modifications=(
            "Especialización para cables subterráneos concéntricos: usa coordenadas locales "
            "normalizadas (cartesianas con distancia log-radial en suelo, polares periódicas "
            "(rho, cos theta, sin theta) en coronas circulares) y amplitud adimensional Q*L^2/k."
        ),
        supports_exact_constraints=False,
        supports_mixed_flux=False,
        tags=("baseline", "subdomain", "strong-form", "thesis-retained"),
    ),
)
class M1SubdomainPINN(_OffsetMLPModel):
    """PINN descompuesta por subdominio ``M1`` con subredes independientes por material."""

    model_kind = "baseline_subdomain"

    def __init__(
        self,
        problem: ThermalProblem2D,
        architecture: str = "M1",
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
            partition="region",
            formulation="strong",
            input_normalization=input_normalization,
        )
        for region in problem.regions:
            in_features = problem.input_size(region, "region")
            self.networks.append(build_mlp(in_features, self.width, self.depth, 1, self.activation))
            self.offsets.append(nn.Parameter(torch.zeros((), dtype=torch.float32)))

    def field(self, region: RegionSpec, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        features = self.input_coordinates(region, xy)
        amplitude = self.problem.amplitude(region, "region")
        temp = self._raw_with_offset(self.networks[region.id], self.offsets[region.id], features, amplitude, channel=0)
        return temp, None


@ModelRegistry.register(
    "M2",
    ModelMetadata(
        name="M2",
        family="mixed",
        partition="region",
        formulation="mixed",
        authors="Gladstone, Nabian, Viana & Meidani (FO-PINN)",
        year=2022,
        doi="arXiv:2211.09321",
        summary=(
            "Formulación mixta de primer orden por subdominio: cada subred predice de forma "
            "simultánea temperatura T y vector de flujo térmico q = (qx, qy), evitando segundas derivadas."
        ),
        modifications=(
            "Se acopla la ley constitutiva adimensional q + k*grad(T) = 0 mediante las escalas "
            "locales de cada material y se impone continuidad tanto de temperatura como de flujo "
            "normal en todas las interfaces entre capas de cable y suelo."
        ),
        supports_exact_constraints=False,
        supports_mixed_flux=True,
        tags=("mixed", "first-order", "subdomain", "thesis-retained"),
    ),
)
class M2MixedFluxPINN(_OffsetMLPModel):
    """PINN mixta ``M2`` que aproxima temperatura y vector flujo térmico en primer orden."""

    model_kind = "mixed_subdomain"

    def __init__(
        self,
        problem: ThermalProblem2D,
        architecture: str = "M2",
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
            partition="region",
            formulation="mixed",
            input_normalization=input_normalization,
        )
        for region in problem.regions:
            in_features = problem.input_size(region, "region")
            self.networks.append(build_mlp(in_features, self.width, self.depth, 3, self.activation))
            self.offsets.append(nn.Parameter(torch.zeros((), dtype=torch.float32)))

    def field(self, region: RegionSpec, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        features = self.input_coordinates(region, xy)
        raw = self.networks[region.id](features)
        amplitude = self.problem.amplitude(region, "region")
        flux_scale = self.problem.scales(region)["flux"]
        temp = self.problem.T_ref + self.problem.T_scale * self.offsets[region.id] + amplitude * raw[:, 0:1]
        flux = flux_scale * raw[:, 1:3]
        return temp, flux
