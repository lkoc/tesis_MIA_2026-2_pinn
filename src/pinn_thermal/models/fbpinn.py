"""Finite Basis Physics-Informed Neural Network (``FBPINN``) para conducción térmica 2D.

Implementación basada en Moseley, Markham & Nissen-Meyer (2023) usando ventanas
solapadas tipo partición de la unidad en subdominios de suelo/material y subredes
independientes sobre regiones de cable o capas diferenciadas.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from pinn_thermal.core.base_model import BasePINNModel, ModelMetadata, ModelRegistry, build_mlp
from pinn_thermal.core.problem import RegionSpec, ThermalProblem2D


@ModelRegistry.register(
    "FBPINN",
    ModelMetadata(
        name="FBPINN",
        family="domain_decomposition",
        partition="region",
        formulation="strong",
        authors="Moseley, Markham & Nissen-Meyer (FBPINN)",
        year=2023,
        doi="10.1007/s10444-023-10065-9",
        summary=(
            "Red neuronal de base finita (Finite Basis PINN): descompone regiones extensas en "
            "ventanas espaciales solapadas suavemente ponderadas mediante partición de la unidad."
        ),
        modifications=(
            "Las regiones rectangulares o poligonales de suelo se particionan en una grilla "
            "2x2 de ventanas coseno solapadas con normalización local en [-1, 1], mientras "
            "las capas circulares de cable conservan sus subredes dedicadas."
        ),
        supports_exact_constraints=False,
        supports_mixed_flux=False,
        tags=("domain-decomposition", "partition-of-unity", "overlapping-windows"),
    ),
)
class FBPINNPINN(BasePINNModel):
    """``FBPINN`` con partición de la unidad por ventanas solapadas y subredes locales."""

    model_kind = "fbpinn"

    def __init__(
        self,
        problem: ThermalProblem2D,
        architecture: str = "FBPINN",
        width: int = 32,
        depth: int = 3,
        activation: str = "tanh",
        input_normalization: str | None = None,
        overlap: float = 0.25,
        grid_nx: int = 2,
        grid_ny: int = 2,
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
        self.overlap = float(overlap)
        self.grid_nx = max(1, int(grid_nx))
        self.grid_ny = max(1, int(grid_ny))

        self.networks = nn.ModuleList()
        self.offsets = nn.ParameterList([nn.Parameter(torch.zeros((), dtype=torch.float32)) for _ in problem.regions])
        self._region_windows: dict[int, list[tuple[int, tuple[float, float, float, float] | None]]] = {}
        self._subnet_keys: list[str] = []

        for region in problem.regions:
            in_features = problem.input_size(region, "region")
            if region.shape.shape_type == "rectangle" and (self.grid_nx > 1 or self.grid_ny > 1):
                x0, x1, y0, y1 = region.shape.bounds  # type: ignore[attr-defined]
                dx = (x1 - x0) / self.grid_nx
                dy = (y1 - y0) / self.grid_ny
                windows: list[tuple[int, tuple[float, float, float, float] | None]] = []
                for ix in range(self.grid_nx):
                    for iy in range(self.grid_ny):
                        wx0 = x0 + max(0.0, ix - self.overlap) * dx
                        wx1 = x0 + min(float(self.grid_nx), ix + 1.0 + self.overlap) * dx
                        wy0 = y0 + max(0.0, iy - self.overlap) * dy
                        wy1 = y0 + min(float(self.grid_ny), iy + 1.0 + self.overlap) * dy
                        net_idx = len(self.networks)
                        self.networks.append(build_mlp(in_features, self.width, self.depth, 1, self.activation))
                        self._subnet_keys.append(f"fbpinn:{region.name}:tile_{ix}_{iy}")
                        windows.append((net_idx, (wx0, wx1, wy0, wy1)))
                self._region_windows[region.id] = windows
            else:
                net_idx = len(self.networks)
                self.networks.append(build_mlp(in_features, self.width, self.depth, 1, self.activation))
                self._subnet_keys.append(f"region:{region.name}")
                self._region_windows[region.id] = [(net_idx, None)]

    @staticmethod
    def _window_weight(xy: torch.Tensor, box: tuple[float, float, float, float]) -> torch.Tensor:
        x0, x1, y0, y1 = box
        ux = 2.0 * (xy[:, 0:1] - x0) / max(x1 - x0, 1e-12) - 1.0
        uy = 2.0 * (xy[:, 1:2] - y0) / max(y1 - y0, 1e-12) - 1.0
        wx = torch.cos(0.5 * math.pi * ux.clamp(-1.0, 1.0)).clamp_min(0.0) ** 2
        wy = torch.cos(0.5 * math.pi * uy.clamp(-1.0, 1.0)).clamp_min(0.0) ** 2
        return wx * wy + 1e-6

    @staticmethod
    def _tile_coords(xy: torch.Tensor, box: tuple[float, float, float, float]) -> torch.Tensor:
        x0, x1, y0, y1 = box
        tx = 2.0 * (xy[:, 0:1] - x0) / max(x1 - x0, 1e-12) - 1.0
        ty = 2.0 * (xy[:, 1:2] - y0) / max(y1 - y0, 1e-12) - 1.0
        return torch.cat([tx, ty], dim=1)

    def field(self, region: RegionSpec, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        windows = self._region_windows[region.id]
        amplitude = self.problem.amplitude(region, "region")
        base_features = self.input_coordinates(region, xy)

        if len(windows) == 1 and windows[0][1] is None:
            raw = self.networks[windows[0][0]](base_features)
            temp = self.problem.T_ref + self.problem.T_scale * self.offsets[region.id] + amplitude * raw
            return temp, None

        weights = [self._window_weight(xy, box) for _, box in windows if box is not None]
        weight_sum = sum(weights).clamp_min(1e-12)
        extra_tail = base_features[:, 2:]
        total_raw = xy.new_zeros((len(xy), 1))
        for (net_idx, box), w in zip(windows, weights):
            assert box is not None
            local = self._tile_coords(xy, box)
            if self.input_normalization == "unit_box_v1":
                local = (local + 1.0) * 0.5
            tile_feats = torch.cat([local, extra_tail], dim=1) if extra_tail.shape[1] > 0 else local
            total_raw = total_raw + (w / weight_sum) * self.networks[net_idx](tile_feats)

        temp = self.problem.T_ref + self.problem.T_scale * self.offsets[region.id] + amplitude * total_raw
        return temp, None

    def subnetwork_map(self) -> dict[str, nn.Module]:
        return {key: net for key, net in zip(self._subnet_keys, self.networks)}
