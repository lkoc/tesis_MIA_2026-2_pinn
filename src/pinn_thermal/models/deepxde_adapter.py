"""Adaptador opcional para usar redes ``DeepXDE`` (PFNN / FNN) dentro del ecosistema ``PINN_V4``.

Permite que las arquitecturas de ``DeepXDE`` operen sobre cualquier ``ThermalProblem2D``
JSON de esta librería sin duplicar lógica de evaluación ni artefactos. Si el paquete
opcional ``deepxde`` no está instalado en la máquina, el resto de ``pinn_thermal``
sigue funcionando con normalidad y solo falla si se solicita explícitamente un modelo DeepXDE.
"""

from __future__ import annotations

import torch
from torch import nn

from pinn_thermal.core.base_model import BasePINNModel, ModelMetadata, ModelRegistry
from pinn_thermal.core.problem import RegionSpec, ThermalProblem2D


def _require_deepxde():
    try:
        import deepxde as dde  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "El paquete opcional 'deepxde' no está instalado. Instálelo con "
            "'pip install deepxde' o utilice los modelos nativos PyTorch (M1, CUSPR, M1_EXACT, etc.)."
        ) from exc
    return dde


class DeepXDEModelAdapter(BasePINNModel):
    """Envuelve una red ``dde.nn.PFNN`` o ``dde.nn.FNN`` bajo la interfaz ``BasePINNModel``."""

    model_kind = "deepxde_adapter"

    def __init__(
        self,
        problem: ThermalProblem2D,
        architecture: str = "DeepXDE_PFNN",
        width: int = 32,
        depth: int = 3,
        activation: str = "tanh",
        input_normalization: str | None = None,
        internal_model: str = "PFNN",
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
            input_normalization=input_normalization or "region_coordinates",
        )
        dde = _require_deepxde()
        self.internal_model = str(internal_model)
        count = len(problem.regions)
        self._feature_sizes = [problem.input_size(region, "region") for region in problem.regions]
        self._feature_width = max(self._feature_sizes)

        layer_sizes = [self._feature_width * count] + (
            [[self.width] * count for _ in range(self.depth)]
            if self.internal_model == "PFNN" and count > 1
            else [self.width] * self.depth
        ) + [count]

        if self.internal_model == "PFNN" and count > 1:
            self.network = dde.nn.PFNN(layer_sizes, self.activation, "Glorot normal")
        else:
            self.network = dde.nn.FNN(layer_sizes, self.activation, "Glorot normal")
        self.offsets = nn.ParameterList([nn.Parameter(torch.zeros((), dtype=torch.float32)) for _ in problem.regions])

    def _encoded_input(self, region: RegionSpec, xy: torch.Tensor) -> torch.Tensor:
        blocks = []
        for reg, size in zip(self.problem.regions, self._feature_sizes):
            if reg.id == region.id:
                feats = self.input_coordinates(reg, xy)
                if size < self._feature_width:
                    feats = nn.functional.pad(feats, (0, self._feature_width - size))
            else:
                feats = xy.new_zeros((len(xy), self._feature_width))
            blocks.append(feats)
        return torch.cat(blocks, dim=1)

    def field(self, region: RegionSpec, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        raw = self.network(self._encoded_input(region, xy))[:, region.id:region.id + 1]
        amplitude = self.problem.amplitude(region, "region")
        temp = self.problem.T_ref + self.problem.T_scale * self.offsets[region.id] + amplitude * raw
        return temp, None

    def subnetwork_map(self) -> dict[str, nn.Module]:
        return {"global": self.network}


ModelRegistry.register_entry(
    "DeepXDE_PFNN",
    ModelMetadata(
        name="DeepXDE_PFNN",
        family="external_adapter",
        partition="region",
        formulation="strong",
        authors="Lu, Meng, Mao & Karniadakis (DeepXDE)",
        year=2021,
        doi="10.1137/19M1274067",
        summary="Adaptador opcional que ejecuta PFNN paralela por subdominio usando el backend PyTorch de DeepXDE.",
        modifications="Conecta entradas normalizadas por región de ThermalProblem2D al evaluador unificado PINN_V4.",
        supports_exact_constraints=False,
        supports_mixed_flux=False,
        tags=("deepxde", "optional"),
    ),
    lambda problem, architecture="DeepXDE_PFNN", width=32, depth=3, activation="tanh", input_normalization=None, **_kw: DeepXDEModelAdapter(
        problem=problem,
        architecture=architecture,
        width=width,
        depth=depth,
        activation=activation,
        input_normalization=input_normalization,
        internal_model="PFNN",
    ),
)

ModelRegistry.register_entry(
    "DeepXDE_FNN",
    ModelMetadata(
        name="DeepXDE_FNN",
        family="external_adapter",
        partition="region",
        formulation="strong",
        authors="Lu, Meng, Mao & Karniadakis (DeepXDE)",
        year=2021,
        doi="10.1137/19M1274067",
        summary="Adaptador opcional que ejecuta una red FNN compartida mediante DeepXDE.",
        modifications="Conecta ThermalProblem2D con FNN multicabezal de DeepXDE preservando compatibilidad de artefactos.",
        supports_exact_constraints=False,
        supports_mixed_flux=False,
        tags=("deepxde", "optional"),
    ),
    lambda problem, architecture="DeepXDE_FNN", width=32, depth=3, activation="tanh", input_normalization=None, **_kw: DeepXDEModelAdapter(
        problem=problem,
        architecture=architecture,
        width=width,
        depth=depth,
        activation=activation,
        input_normalization=input_normalization,
        internal_model="FNN",
    ),
)
