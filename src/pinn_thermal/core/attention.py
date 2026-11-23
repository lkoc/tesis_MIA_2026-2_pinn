"""Mecanismos adaptativos de ponderación puntual por residuo (RBA y SA).

Solo deben utilizarse con optimizadores de primer orden como Adam, ya que alteran
dinámicamente los pesos de los puntos de colocación entre iteraciones.

Referencias originales:
- **RBA** (*Residual-Based Attention*): Anagnostopoulos, Toscano, Stergiopulos y
  Karniadakis (2024), DOI ``10.1016/j.cma.2024.116805``.
- **SA** (*Self-Adaptive Weights*): McClenny y Braga-Neto (2023),
  ``Journal of Computational Physics, 474``, 111722 (preprint ``arXiv:2009.04544``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from typing import Any

import torch

ATTENTION_KINDS: tuple[str, ...] = ("rba", "sa")


class BaseAttention(ABC):
    """Interfaz base de ponderación puntual adaptativa por región."""

    kind: str = ""

    def __init__(
        self,
        sizes: dict[str, int],
        rate: float = 0.01,
        decay: float = 0.999,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        self.sizes = dict(sizes)
        self.rate = float(rate)
        self.decay = float(decay)
        start = 0.0 if self.kind == "rba" else 1.0
        self.state: dict[str, torch.Tensor] = {
            key: torch.full((int(n), 1), start, dtype=dtype, device=device)
            for key, n in sizes.items()
        }

    @abstractmethod
    def weight(self, key: str) -> torch.Tensor:
        """Devuelve el tensor de pesos efectivos para la región ``key``."""

    @abstractmethod
    def update(self, key: str, squared_residual: torch.Tensor) -> None:
        """Actualiza los pesos de forma cerrada a partir del residuo al cuadrado sin gradiente."""

    def summary(self) -> dict[str, dict[str, float]]:
        return {
            key: {
                "min": float(self.weight(key).min()),
                "max": float(self.weight(key).max()),
                "mean": float(self.weight(key).mean()),
            }
            for key in self.state
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "rate": self.rate,
            "decay": self.decay,
            "state": {k: v.detach().cpu().clone() for k, v in self.state.items()},
        }

    def load_state_dict(self, payload: dict[str, Any], ref_tensor: torch.Tensor | None = None) -> None:
        if payload.get("kind") != self.kind:
            raise ValueError(f"Atención incompatible: esperado {self.kind!r}, recibido {payload.get('kind')!r}.")
        self.rate = float(payload["rate"])
        self.decay = float(payload["decay"])
        if ref_tensor is not None:
            self.state = {k: v.to(ref_tensor).clone() for k, v in payload["state"].items()}
        else:
            self.state = copy.deepcopy(payload["state"])


class ResidualBasedAttention(BaseAttention):
    """Atención residual RBA: ``lam <- decay * lam + rate * |r| / max(|r|)``, peso ``1 + lam``."""

    kind = "rba"

    def weight(self, key: str) -> torch.Tensor:
        return 1.0 + self.state[key]

    def update(self, key: str, squared_residual: torch.Tensor) -> None:
        r2 = squared_residual.detach().reshape(-1, 1)
        r = torch.sqrt(r2.clamp_min(0.0))
        denom = r.max().clamp_min(torch.finfo(r.dtype).tiny)
        self.state[key] = self.decay * self.state[key] + self.rate * r / denom


class SelfAdaptiveAttention(BaseAttention):
    """Pesos autoadaptativos normalizados a media unitaria para preservar la escala de la PDE."""

    kind = "sa"

    def weight(self, key: str) -> torch.Tensor:
        st = self.state[key]
        return st / st.mean().clamp_min(torch.finfo(st.dtype).tiny)

    def update(self, key: str, squared_residual: torch.Tensor) -> None:
        r2 = squared_residual.detach().reshape(-1, 1)
        denom = r2.mean().clamp_min(torch.finfo(r2.dtype).tiny)
        self.state[key] = self.state[key] + self.rate * r2 / denom


def build_attention(
    kind: str,
    sizes: dict[str, int],
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
    rate: float | None = None,
    decay: float = 0.999,
) -> BaseAttention:
    """Fábrica de instancias de atención adaptativa (``'rba'`` o ``'sa'``)."""
    cls_map: dict[str, type[BaseAttention]] = {
        "rba": ResidualBasedAttention,
        "sa": SelfAdaptiveAttention,
    }
    if kind not in cls_map:
        raise ValueError(f"Tipo de atención desconocido {kind!r}; elija uno de {ATTENTION_KINDS}.")
    return cls_map[kind](
        sizes=sizes,
        rate=0.01 if rate is None else float(rate),
        decay=decay,
        dtype=dtype,
        device=device,
    )
