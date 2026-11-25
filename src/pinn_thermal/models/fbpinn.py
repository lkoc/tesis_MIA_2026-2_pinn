"""Finite Basis Physics-Informed Neural Network (``FBPINN``) para conducción térmica 2D.

Implementación basada en Moseley, Markham & Nissen-Meyer (2023) usando ventanas
solapadas tipo partición de la unidad en subdominios de suelo/material y subredes
independientes sobre regiones de cable o capas diferenciadas.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from pinn_thermal.core.base_model import BasePINNModel, ModelMetadata, ModelRegistry, build_mlp
from pinn_thermal.core.problem import RegionSpec, ThermalProblem2D


@dataclass(frozen=True)
class _TileWindow:
    """Especificación de ventana tensorizada y caja local de normalización para ``FBPINN``."""

    net_idx: int
    box: tuple[float, float, float, float] | None
    x_left: tuple[float, float] | None = None
    x_right: tuple[float, float] | None = None
    y_left: tuple[float, float] | None = None
    y_right: tuple[float, float] | None = None


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
            "2x2 de ventanas C2 tipo partición de la unidad (smootherstep en solapamientos "
            "interiores y valor 1 en bordes globales) con normalización local en [-1, 1], "
            "mientras las capas circulares de cable conservan sus subredes dedicadas."
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
        self._region_windows: dict[int, list[_TileWindow]] = {}
        self._subnet_keys: list[str] = []

        alpha = min(max(self.overlap, 1e-3), 0.5)
        for region in problem.regions:
            in_features = problem.input_size(region, "region")
            if region.shape.shape_type == "rectangle" and (self.grid_nx > 1 or self.grid_ny > 1):
                x0, x1, y0, y1 = region.shape.bounds  # type: ignore[attr-defined]
                dx = (x1 - x0) / self.grid_nx
                dy = (y1 - y0) / self.grid_ny
                windows: list[_TileWindow] = []
                for ix in range(self.grid_nx):
                    wx0 = x0 if ix == 0 else x0 + (ix - alpha) * dx
                    wx1 = x1 if ix == self.grid_nx - 1 else x0 + (ix + 1.0 + alpha) * dx
                    x_left = (x0 + (ix - alpha) * dx, x0 + (ix + alpha) * dx) if ix > 0 else None
                    x_right = (
                        (x0 + (ix + 1.0 - alpha) * dx, x0 + (ix + 1.0 + alpha) * dx)
                        if ix < self.grid_nx - 1
                        else None
                    )
                    for iy in range(self.grid_ny):
                        wy0 = y0 if iy == 0 else y0 + (iy - alpha) * dy
                        wy1 = y1 if iy == self.grid_ny - 1 else y0 + (iy + 1.0 + alpha) * dy
                        y_left = (y0 + (iy - alpha) * dy, y0 + (iy + alpha) * dy) if iy > 0 else None
                        y_right = (
                            (y0 + (iy + 1.0 - alpha) * dy, y0 + (iy + 1.0 + alpha) * dy)
                            if iy < self.grid_ny - 1
                            else None
                        )
                        net_idx = len(self.networks)
                        self.networks.append(build_mlp(in_features, self.width, self.depth, 1, self.activation))
                        self._subnet_keys.append(f"fbpinn:{region.name}:tile_{ix}_{iy}")
                        windows.append(
                            _TileWindow(
                                net_idx=net_idx,
                                box=(wx0, wx1, wy0, wy1),
                                x_left=x_left,
                                x_right=x_right,
                                y_left=y_left,
                                y_right=y_right,
                            )
                        )
                self._region_windows[region.id] = windows
            else:
                net_idx = len(self.networks)
                self.networks.append(build_mlp(in_features, self.width, self.depth, 1, self.activation))
                self._subnet_keys.append(f"region:{region.name}")
                self._region_windows[region.id] = [_TileWindow(net_idx=net_idx, box=None)]

    @staticmethod
    def _smootherstep(s: torch.Tensor) -> torch.Tensor:
        """Polinomio C2 monótono en [0, 1] con primera y segunda derivada nulas en los extremos."""
        t = s.clamp(0.0, 1.0)
        return t * t * t * (10.0 + t * (-15.0 + 6.0 * t))

    @classmethod
    def _axis_window(
        cls,
        coord: torch.Tensor,
        left: tuple[float, float] | None,
        right: tuple[float, float] | None,
    ) -> torch.Tensor:
        w = torch.ones_like(coord)
        if left is not None:
            a, b = left
            w = w * cls._smootherstep((coord - a) / max(b - a, 1e-12))
        if right is not None:
            a, b = right
            w = w * (1.0 - cls._smootherstep((coord - a) / max(b - a, 1e-12)))
        return w

    @classmethod
    def _window_weight(cls, xy: torch.Tensor, win: _TileWindow) -> torch.Tensor:
        return cls._axis_window(xy[:, 0:1], win.x_left, win.x_right) * cls._axis_window(
            xy[:, 1:2], win.y_left, win.y_right
        )

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

        if len(windows) == 1 and windows[0].box is None:
            raw = self.networks[windows[0].net_idx](base_features)
            temp = self.problem.T_ref + self.problem.T_scale * self.offsets[region.id] + amplitude * raw
            return temp, None

        weights = [self._window_weight(xy, win) for win in windows if win.box is not None]
        weight_sum = sum(weights).clamp_min(1e-12)
        extra_tail = base_features[:, 2:]
        total_raw = xy.new_zeros((len(xy), 1))
        for win, w in zip(windows, weights):
            assert win.box is not None
            local = self._tile_coords(xy, win.box)
            if self.input_normalization == "unit_box_v1":
                local = (local + 1.0) * 0.5
            tile_feats = torch.cat([local, extra_tail], dim=1) if extra_tail.shape[1] > 0 else local
            total_raw = total_raw + (w / weight_sum) * self.networks[win.net_idx](tile_feats)

        temp = self.problem.T_ref + self.problem.T_scale * self.offsets[region.id] + amplitude * total_raw
        return temp, None

    def subnetwork_map(self) -> dict[str, nn.Module]:
        return {key: net for key, net in zip(self._subnet_keys, self.networks)}
