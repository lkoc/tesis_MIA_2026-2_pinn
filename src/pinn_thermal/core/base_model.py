"""Clase base abstracta de modelos PINN, bloques MLP y registro extensible de arquitecturas.

Define la Capa 1 (Core) sobre la que se construye cada familia especializada en
``pinn_thermal.models``. Agregar un nuevo modelo en el futuro consiste simplemente en:
1. Heredar de :class:`BasePINNModel` (o una subclase existente).
2. Implementar ``field(region, xy)`` (y ``compatibility_profile`` / ``subnetwork_map`` si añade subredes propias).
3. Decorar la clase o registrarla con :meth:`ModelRegistry.register`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from pinn_thermal.core.physics import derived_flux
from pinn_thermal.core.problem import RegionSpec, ThermalProblem2D

ACTIVATIONS: dict[str, type[nn.Module]] = {
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
}


class SinActivation(nn.Module):
    """Activación sinusoidal (útil en redes tipo SIREN para soluciones oscilatorias)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


def build_mlp(
    in_features: int,
    width: int,
    depth: int,
    out_features: int,
    activation: str = "tanh",
    init_mode: str = "xavier_normal",
) -> nn.Sequential:
    """Construye una red perceptrón multicapa (MLP) totalmente conectada.

    La última capa es siempre una transformación lineal sin activación ``nn.Linear``
    para permitir proyección por mínimos cuadrados (``readout``) sobre sus pesos.
    """
    act_key = activation.lower()
    if act_key == "sin":
        act_cls: Callable[[], nn.Module] = SinActivation
    elif act_key in ACTIVATIONS:
        act_cls = ACTIVATIONS[act_key]
    else:
        raise ValueError(f"Activación no soportada {activation!r}; elija entre {sorted([*ACTIVATIONS, 'sin'])}.")

    init_fn = nn.init.xavier_normal_ if init_mode == "xavier_normal" else nn.init.xavier_uniform_
    layers: list[nn.Module] = []
    size = int(in_features)
    for _ in range(int(depth)):
        linear = nn.Linear(size, int(width))
        init_fn(linear.weight)
        nn.init.zeros_(linear.bias)
        layers.extend([linear, act_cls()])
        size = int(width)
    last = nn.Linear(size, int(out_features))
    init_fn(last.weight)
    nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class ModelMetadata:
    """Ficha documental y bibliográfica de una arquitectura PINN registrada."""

    name: str
    family: str
    partition: str
    formulation: str
    authors: str
    year: int | None
    doi: str = ""
    summary: str = ""
    modifications: str = ""
    supports_exact_constraints: bool = False
    supports_mixed_flux: bool = False
    tags: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "partition": self.partition,
            "formulation": self.formulation,
            "authors": self.authors,
            "year": self.year,
            "doi": self.doi,
            "summary": self.summary,
            "modifications": self.modifications,
            "supports_exact_constraints": self.supports_exact_constraints,
            "supports_mixed_flux": self.supports_mixed_flux,
            "tags": list(self.tags),
        }


class BasePINNModel(nn.Module, ABC):
    """Interfaz común orientada a objetos para todos los modelos PINN del núcleo térmico."""

    model_kind: str = "baseline"
    exact_constraints: bool = False
    exact_boundary: bool = False

    def __init__(
        self,
        problem: ThermalProblem2D,
        architecture: str,
        width: int = 32,
        depth: int = 3,
        activation: str = "tanh",
        partition: str = "region",
        formulation: str = "strong",
        input_normalization: str = "region_coordinates",
    ) -> None:
        super().__init__()
        self.problem: ThermalProblem2D = problem
        self.architecture: str = str(architecture)
        self.width: int = int(width)
        self.depth: int = int(depth)
        self.activation: str = str(activation)
        self.partition: str = str(partition)
        self.formulation: str = str(formulation)
        self.mixed: bool = self.formulation == "mixed"
        self.input_normalization: str = str(input_normalization)

    def input_coordinates(self, region: RegionSpec, xy: torch.Tensor) -> torch.Tensor:
        """Codifica coordenadas físicas según la política de normalización del modelo."""
        if self.input_normalization == "unit_box_v1":
            return self.problem.unit_coordinates(region, xy, self.partition)
        return self.problem.coordinates(region, xy, self.partition)

    @abstractmethod
    def field(self, region: RegionSpec, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Evalúa ``(temperatura_C, flujo_o_None)`` dentro de ``region`` sobre los puntos ``xy``."""

    def temperature(self, xy: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Evalúa la temperatura escalar para puntos etiquetados por su región propietaria."""
        out = torch.empty((len(xy), 1), dtype=xy.dtype, device=xy.device)
        for region in self.problem.regions:
            mask = labels == region.id
            if bool(torch.any(mask)):
                out[mask] = self.field(region, xy[mask])[0]
        return out

    def predict(
        self,
        xy: np.ndarray,
        with_flux: bool = False,
        chunk_size: int = 20000,
    ) -> dict[str, np.ndarray]:
        """Inferencia uniforme desde NumPy sobre cualquier dispositivo (CPU/GPU/XPU/NPU).

        Devuelve un diccionario con ``T``, ``region_ids`` y opcionalmente ``q`` (flujo derivado).
        """
        pts = np.asarray(xy, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or not np.isfinite(pts).all():
            raise ValueError("Las coordenadas de inferencia deben formar un arreglo finito (N, 2) en metros.")
        labels = np.asarray(self.problem.owner_index(pts), dtype=np.int64)
        if np.any(labels < 0):
            raise ValueError("Existen puntos de inferencia fuera del dominio físico del problema.")

        ref_param = next(self.parameters())
        temps = np.empty(len(pts), dtype=float)
        flux_arr = np.empty((len(pts), 2), dtype=float) if with_flux else None

        was_training = self.training
        self.eval()
        try:
            with torch.set_grad_enabled(with_flux):
                for region in self.problem.regions:
                    indices = np.flatnonzero(labels == region.id)
                    if not len(indices):
                        continue
                    for start in range(0, len(indices), chunk_size):
                        batch = indices[start:start + chunk_size]
                        coords = torch.as_tensor(
                            pts[batch], dtype=ref_param.dtype, device=ref_param.device
                        ).requires_grad_(with_flux)
                        t_pred, q_pred = self.field(region, coords)
                        temps[batch] = t_pred.detach().cpu().numpy().ravel()
                        if with_flux and flux_arr is not None:
                            if q_pred is not None:
                                q_out = q_pred
                            else:
                                k = self.problem.conductivity(region, coords)
                                q_out = derived_flux(t_pred, coords, k)
                            flux_arr[batch] = q_out.detach().cpu().numpy()
        finally:
            if was_training:
                self.train()

        result: dict[str, np.ndarray] = {"T": temps, "region_ids": labels}
        if flux_arr is not None:
            result["q"] = flux_arr
        return result

    def parameter_count(self) -> int:
        """Número total de parámetros entrenables del modelo."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def subnetwork_map(self) -> dict[str, nn.Module]:
        """Mapa simbólico ``clave -> subred`` para transferencia de pesos y readout.

        Las claves por defecto usan ``region:<nombre>`` (para partición por subdominio)
        o ``global`` (para redes compartidas), más entradas auxiliares si existen.
        """
        mapping: dict[str, nn.Module] = {}
        networks = getattr(self, "networks", None)
        if isinstance(networks, nn.ModuleList) and len(networks) > 0:
            if self.partition == "global" and len(networks) == 1:
                mapping["global"] = networks[0]
            elif len(networks) == len(self.problem.regions):
                for region, net in zip(self.problem.regions, networks):
                    mapping[f"region:{region.name}"] = net
            else:
                for idx, net in enumerate(networks):
                    mapping[f"net:{idx}"] = net
        return mapping

    def compatibility_profile(self) -> dict[str, Any]:
        """Perfil estructurado de compatibilidad para comparar con otros artefactos/modelos."""
        subnet_shapes: dict[str, dict[str, Any]] = {}
        for key, mod in self.subnetwork_map().items():
            linears = [m for m in mod.modules() if isinstance(m, nn.Linear)]
            if linears:
                subnet_shapes[key] = {
                    "in_features": int(linears[0].in_features),
                    "out_features": int(linears[-1].out_features),
                    "shape_signature": [(int(l.in_features), int(l.out_features)) for l in linears],
                }
        return {
            "architecture": self.architecture,
            "model_kind": self.model_kind,
            "partition": self.partition,
            "formulation": self.formulation,
            "mixed": self.mixed,
            "width": self.width,
            "depth": self.depth,
            "activation": self.activation,
            "input_normalization": self.input_normalization,
            "exact_constraints": self.exact_constraints,
            "exact_boundary": self.exact_boundary,
            "domain_shape": self.problem.domain_shape,
            "region_names": [r.name for r in self.problem.regions],
            "region_kinds": {r.name: r.kind for r in self.problem.regions},
            "subnets": subnet_shapes,
        }


class ModelRegistry:
    """Registro central basado en clases para arquitecturas PINN extensibles."""

    _factories: dict[str, Callable[..., BasePINNModel]] = {}
    _metadata: dict[str, ModelMetadata] = {}

    @classmethod
    def register(
        cls,
        name: str,
        metadata: ModelMetadata,
        factory: Callable[..., BasePINNModel] | None = None,
    ) -> Callable[[Any], Any]:
        """Registra una clase o función fábrica bajo ``name`` y guarda su ficha teórica."""
        def _decorator(obj: Any) -> Any:
            cls._factories[name] = factory if factory is not None else obj
            cls._metadata[name] = metadata
            return obj

        if factory is not None:
            cls._factories[name] = factory
            cls._metadata[name] = metadata
        return _decorator

    @classmethod
    def register_entry(
        cls,
        name: str,
        metadata: ModelMetadata,
        factory: Callable[..., BasePINNModel],
    ) -> None:
        """Registra directamente una entrada sin sintaxis de decorador."""
        cls._factories[name] = factory
        cls._metadata[name] = metadata

    @classmethod
    def list_models(cls) -> list[str]:
        """Lista ordenada de todas las arquitecturas registradas."""
        return sorted(cls._factories)

    @classmethod
    def get_metadata(cls, name: str) -> ModelMetadata:
        """Devuelve la ficha documental del modelo solicitado."""
        if name not in cls._metadata:
            raise KeyError(f"Modelo PINN desconocido {name!r}; disponibles: {cls.list_models()}.")
        return cls._metadata[name]

    @classmethod
    def describe_all(cls) -> list[dict[str, Any]]:
        """Resumen serializable de todos los modelos registrados."""
        return [cls._metadata[name].to_dict() for name in cls.list_models()]

    @classmethod
    def build(
        cls,
        problem: ThermalProblem2D,
        architecture: str,
        width: int = 32,
        depth: int = 3,
        activation: str = "tanh",
        **kwargs: Any,
    ) -> BasePINNModel:
        """Instancia el modelo registrado con los hiperparámetros declarados."""
        if architecture not in cls._factories:
            raise ValueError(
                f"Arquitectura no registrada {architecture!r}. "
                f"Modelos disponibles: {cls.list_models()}."
            )
        factory = cls._factories[architecture]
        return factory(
            problem=problem,
            architecture=architecture,
            width=width,
            depth=depth,
            activation=activation,
            **kwargs,
        )
