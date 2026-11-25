"""Geometría bidimensional orientada a objetos para regiones materiales y dominios.

Soporta las formas canónicas heredadas de ``pinn-v3-problem-1`` y añade soporte
general para polígonos planos 2D (``polygon``):
- ``RectangleShape`` (``rectangle``)
- ``DiskShape`` (``disk``)
- ``AnnulusShape`` (``annulus``)
- ``PolygonShape`` (``polygon``)

Este módulo es **100 % libre de PyTorch**, operando exclusivamente con NumPy para
compartir exactamente las mismas pruebas de pertenencia, márgenes y muestreo local
entre el núcleo PINN y el adaptador FEniCSx.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

SHAPES: tuple[str, ...] = ("rectangle", "disk", "annulus", "polygon")
RING_FRACTION: float = 0.05
BOUNDARY_FRACTION: float = 1e-3


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


class Shape2D(ABC):
    """Clase base abstracta de geometrías bidimensionales de región."""

    shape_type: str = ""

    @abstractmethod
    def contains(self, x: np.ndarray, y: np.ndarray, tolerance: float = 0.0) -> np.ndarray:
        """Devuelve una máscara booleana de los puntos ``(x, y)`` dentro de la región."""

    @abstractmethod
    def boundary_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Distancia euclidiana aproximada o exacta al borde más cercano de la región."""

    @abstractmethod
    def domain_to_local(self, xy: np.ndarray) -> np.ndarray:
        """Transforma puntos físicos ``(N, 2)`` a coordenadas normalizadas locales."""

    @abstractmethod
    def local_to_domain(self, local: np.ndarray) -> np.ndarray:
        """Transforma coordenadas normalizadas locales a coordenadas físicas ``(N, 2)``."""

    @abstractmethod
    def characteristic_size(self) -> float:
        """Longitud característica mínima interior (lado, radio o espesor anular)."""

    @abstractmethod
    def area(self) -> float:
        """Área geométrica exacta de la región sin descontar solapamientos de mayor prioridad."""

    @abstractmethod
    def interior_margin(self, span: float, factor: float = 1e-7) -> float:
        """Margen de exclusión respecto a fronteras e interfaces escalado localmente."""

    @abstractmethod
    def sample_local(self, count: int, rng: np.random.Generator, margin: float = 0.0) -> np.ndarray:
        """Genera ``count`` muestras uniformes en área dentro de la región."""

    @abstractmethod
    def inside_bounds(self, bounds: tuple[float, float, float, float] | list[float]) -> bool:
        """Comprueba si toda la geometría permanece dentro de la caja ``[x0, x1, y0, y1]``."""

    @abstractmethod
    def boundary_curves(self, n_per_curve: int) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """Devuelve tuplas ``(xy, normal_exterior, longitud_fisica)`` sobre cada tramo de borde."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serializa la geometría a su diccionario JSON canónico."""

    def ring_offset(self) -> float:
        """Desplazamiento normal de los anillos de interfaz respecto a la curva compartida."""
        return RING_FRACTION * self.characteristic_size()

    def boundary_offset(self) -> float:
        """Desplazamiento normal de la segunda fila interior en fronteras externas."""
        return BOUNDARY_FRACTION * self.characteristic_size()

    @staticmethod
    def from_dict(spec: dict[str, Any], where: str = "shape") -> Shape2D:
        """Construye la instancia ``Shape2D`` correspondiente a un diccionario JSON validado."""
        if isinstance(spec, Shape2D):
            return spec
        if not isinstance(spec, dict) or spec.get("type") not in SHAPES:
            raise ValueError(f"{where}: shape.type debe ser uno de {SHAPES}.")
        kind = spec["type"]
        if kind == "rectangle":
            if set(spec) != {"type", "bounds"} or len(spec["bounds"]) != 4:
                raise ValueError(f"{where}: un rectángulo solo debe declarar type y bounds [x0, x1, y0, y1].")
            if not all(_is_finite_number(v) for v in spec["bounds"]):
                raise ValueError(f"{where}: los límites del rectángulo deben ser finitos.")
            x0, x1, y0, y1 = (float(v) for v in spec["bounds"])
            return RectangleShape((x0, x1, y0, y1))
        if kind == "disk":
            if set(spec) != {"type", "center", "radius_m"}:
                raise ValueError(f"{where}: un disco solo debe declarar type, center y radius_m.")
            if not isinstance(spec["center"], (list, tuple)) or len(spec["center"]) != 2:
                raise ValueError(f"{where}: center debe ser [cx, cy].")
            if not all(_is_finite_number(v) for v in spec["center"]) or not _is_finite_number(spec["radius_m"]):
                raise ValueError(f"{where}: centro y radio del disco deben ser finitos.")
            return DiskShape((float(spec["center"][0]), float(spec["center"][1])), float(spec["radius_m"]))
        if kind == "annulus":
            if set(spec) != {"type", "center", "inner_radius_m", "outer_radius_m"}:
                raise ValueError(f"{where}: un anillo debe declarar center, inner_radius_m y outer_radius_m.")
            if not isinstance(spec["center"], (list, tuple)) or len(spec["center"]) != 2:
                raise ValueError(f"{where}: center debe ser [cx, cy].")
            if (
                not all(_is_finite_number(v) for v in spec["center"])
                or not _is_finite_number(spec["inner_radius_m"])
                or not _is_finite_number(spec["outer_radius_m"])
            ):
                raise ValueError(f"{where}: centro y radios del anillo deben ser finitos.")
            return AnnulusShape(
                (float(spec["center"][0]), float(spec["center"][1])),
                float(spec["inner_radius_m"]),
                float(spec["outer_radius_m"]),
            )
        # polygon
        if set(spec) != {"type", "vertices"}:
            raise ValueError(f"{where}: un polígono debe declarar type y vertices [[x0, y0], ...].")
        verts = spec["vertices"]
        if not isinstance(verts, (list, tuple)) or len(verts) < 3:
            raise ValueError(f"{where}: un polígono necesita al menos 3 vértices.")
        parsed = []
        for idx, pt in enumerate(verts):
            if not isinstance(pt, (list, tuple)) or len(pt) != 2 or not all(_is_finite_number(c) for c in pt):
                raise ValueError(f"{where}: vértice {idx} inválido; debe ser [x, y] finito.")
            parsed.append((float(pt[0]), float(pt[1])))
        return PolygonShape(tuple(parsed))


@dataclass(frozen=True)
class RectangleShape(Shape2D):
    """Región rectangular alineada con los ejes ``[x0, x1, y0, y1]``."""

    bounds: tuple[float, float, float, float]
    shape_type: str = "rectangle"

    def __post_init__(self) -> None:
        x0, x1, y0, y1 = self.bounds
        if not (x0 < x1 and y0 < y1):
            raise ValueError(f"Rectángulo degenerado o vacío: {self.bounds}.")

    def contains(self, x: np.ndarray, y: np.ndarray, tolerance: float = 0.0) -> np.ndarray:
        x0, x1, y0, y1 = self.bounds
        return (x >= x0 - tolerance) & (x <= x1 + tolerance) & (y >= y0 - tolerance) & (y <= y1 + tolerance)

    def boundary_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x0, x1, y0, y1 = self.bounds
        return np.minimum.reduce((np.abs(x - x0), np.abs(x1 - x), np.abs(y - y0), np.abs(y1 - y)))

    def domain_to_local(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        x0, x1, y0, y1 = self.bounds
        return np.c_[(xy[:, 0] - x0) / (x1 - x0), (xy[:, 1] - y0) / (y1 - y0)]

    def local_to_domain(self, local: np.ndarray) -> np.ndarray:
        local = np.asarray(local, dtype=float)
        x0, x1, y0, y1 = self.bounds
        return np.c_[x0 + local[:, 0] * (x1 - x0), y0 + local[:, 1] * (y1 - y0)]

    def characteristic_size(self) -> float:
        x0, x1, y0, y1 = self.bounds
        return float(min(x1 - x0, y1 - y0))

    def area(self) -> float:
        x0, x1, y0, y1 = self.bounds
        return float((x1 - x0) * (y1 - y0))

    def interior_margin(self, span: float, factor: float = 1e-7) -> float:
        x0, x1, y0, y1 = self.bounds
        return min(factor * max(float(span), 1.0), 0.1 * min(x1 - x0, y1 - y0))

    def sample_local(self, count: int, rng: np.random.Generator, margin: float = 0.0) -> np.ndarray:
        count = int(count)
        if count < 0:
            raise ValueError("La cantidad de muestras debe ser no negativa.")
        if count == 0:
            return np.empty((0, 2), dtype=float)
        x0, x1, y0, y1 = self.bounds
        mx = min(float(margin) / (x1 - x0), 0.1)
        my = min(float(margin) / (y1 - y0), 0.1)
        local = np.c_[rng.uniform(mx, 1.0 - mx, count), rng.uniform(my, 1.0 - my, count)]
        return self.local_to_domain(local)

    def inside_bounds(self, bounds: tuple[float, float, float, float] | list[float]) -> bool:
        x0, x1, y0, y1 = bounds
        a0, a1, b0, b1 = self.bounds
        return a0 >= x0 and a1 <= x1 and b0 >= y0 and b1 <= y1

    def boundary_curves(self, n_per_curve: int) -> list[tuple[np.ndarray, np.ndarray, float]]:
        x0, x1, y0, y1 = self.bounds
        s = (np.arange(n_per_curve) + 0.5) / n_per_curve
        ones = np.ones(n_per_curve)
        return [
            (np.c_[x0 + (x1 - x0) * s, y0 * ones], np.tile((0.0, -1.0), (n_per_curve, 1)), x1 - x0),
            (np.c_[x0 + (x1 - x0) * s, y1 * ones], np.tile((0.0, 1.0), (n_per_curve, 1)), x1 - x0),
            (np.c_[x0 * ones, y0 + (y1 - y0) * s], np.tile((-1.0, 0.0), (n_per_curve, 1)), y1 - y0),
            (np.c_[x1 * ones, y0 + (y1 - y0) * s], np.tile((1.0, 0.0), (n_per_curve, 1)), y1 - y0),
        ]

    def to_dict(self) -> dict[str, Any]:
        return {"type": "rectangle", "bounds": list(self.bounds)}


@dataclass(frozen=True)
class DiskShape(Shape2D):
    """Región circular sólida (conductor macizo o inclusión circular)."""

    center: tuple[float, float]
    radius_m: float
    shape_type: str = "disk"

    def __post_init__(self) -> None:
        if self.radius_m <= 0:
            raise ValueError(f"El radio del disco debe ser positivo: {self.radius_m}.")

    def contains(self, x: np.ndarray, y: np.ndarray, tolerance: float = 0.0) -> np.ndarray:
        cx, cy = self.center
        return (x - cx) ** 2 + (y - cy) ** 2 <= (self.radius_m + tolerance) ** 2

    def boundary_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        cx, cy = self.center
        return np.abs(np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - self.radius_m)

    def domain_to_local(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        return (xy - np.asarray(self.center, dtype=float)) / self.radius_m

    def local_to_domain(self, local: np.ndarray) -> np.ndarray:
        local = np.asarray(local, dtype=float)
        return np.asarray(self.center, dtype=float) + self.radius_m * local[:, :2]

    def characteristic_size(self) -> float:
        return float(self.radius_m)

    def area(self) -> float:
        return float(math.pi * (self.radius_m ** 2))

    def interior_margin(self, span: float, factor: float = 1e-7) -> float:
        return min(factor * max(float(span), 1.0), 0.1 * self.radius_m)

    def sample_local(self, count: int, rng: np.random.Generator, margin: float = 0.0) -> np.ndarray:
        count = int(count)
        if count < 0:
            raise ValueError("La cantidad de muestras debe ser no negativa.")
        if count == 0:
            return np.empty((0, 2), dtype=float)
        effective_r = self.radius_m - float(margin)
        if effective_r <= 0:
            raise RuntimeError("El disco no conserva área interior tras descontar el margen de interfaz.")
        radial = np.sqrt(rng.uniform(0.0, 1.0, count)) * (effective_r / self.radius_m)
        angle = rng.uniform(0.0, 2.0 * math.pi, count)
        local = np.c_[radial * np.cos(angle), radial * np.sin(angle)]
        return self.local_to_domain(local)

    def inside_bounds(self, bounds: tuple[float, float, float, float] | list[float]) -> bool:
        x0, x1, y0, y1 = bounds
        cx, cy = self.center
        r = self.radius_m
        return cx - r >= x0 and cx + r <= x1 and cy - r >= y0 and cy + r <= y1

    def boundary_curves(self, n_per_curve: int) -> list[tuple[np.ndarray, np.ndarray, float]]:
        center = np.asarray(self.center, dtype=float)
        angle = 2.0 * np.pi * (np.arange(n_per_curve) + 0.5) / n_per_curve
        radial = np.c_[np.cos(angle), np.sin(angle)]
        return [(center + self.radius_m * radial, radial, 2.0 * np.pi * self.radius_m)]

    def to_dict(self) -> dict[str, Any]:
        return {"type": "disk", "center": list(self.center), "radius_m": float(self.radius_m)}


@dataclass(frozen=True)
class AnnulusShape(Shape2D):
    """Región anular concéntrica (aislamiento, pantalla metálica o cubierta exterior)."""

    center: tuple[float, float]
    inner_radius_m: float
    outer_radius_m: float
    shape_type: str = "annulus"

    def __post_init__(self) -> None:
        if not (0.0 < self.inner_radius_m < self.outer_radius_m):
            raise ValueError(
                f"El anillo requiere 0 < inner_radius_m < outer_radius_m; recibido "
                f"({self.inner_radius_m}, {self.outer_radius_m})."
            )

    def contains(self, x: np.ndarray, y: np.ndarray, tolerance: float = 0.0) -> np.ndarray:
        cx, cy = self.center
        r2 = (x - cx) ** 2 + (y - cy) ** 2
        r_in = max(self.inner_radius_m - tolerance, 0.0) ** 2
        r_out = (self.outer_radius_m + tolerance) ** 2
        return (r2 >= r_in) & (r2 <= r_out)

    def boundary_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        cx, cy = self.center
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        return np.minimum(np.abs(r - self.inner_radius_m), np.abs(r - self.outer_radius_m))

    def domain_to_local(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        delta = xy - np.asarray(self.center, dtype=float)
        radius = np.linalg.norm(delta, axis=1)
        angle = np.mod(np.arctan2(delta[:, 1], delta[:, 0]), 2.0 * math.pi)
        radial = (radius - self.inner_radius_m) / (self.outer_radius_m - self.inner_radius_m)
        return np.c_[radial, angle / (2.0 * math.pi)]

    def local_to_domain(self, local: np.ndarray) -> np.ndarray:
        local = np.asarray(local, dtype=float)
        center = np.asarray(self.center, dtype=float)
        radius = self.inner_radius_m + local[:, 0] * (self.outer_radius_m - self.inner_radius_m)
        angle = 2.0 * math.pi * local[:, 1]
        return center + np.c_[radius * np.cos(angle), radius * np.sin(angle)]

    def characteristic_size(self) -> float:
        return float(self.outer_radius_m - self.inner_radius_m)

    def area(self) -> float:
        return float(math.pi * (self.outer_radius_m ** 2 - self.inner_radius_m ** 2))

    def interior_margin(self, span: float, factor: float = 1e-7) -> float:
        thickness = self.outer_radius_m - self.inner_radius_m
        return min(factor * max(float(span), 1.0), 0.25 * thickness)

    def sample_local(self, count: int, rng: np.random.Generator, margin: float = 0.0) -> np.ndarray:
        count = int(count)
        if count < 0:
            raise ValueError("La cantidad de muestras debe ser no negativa.")
        if count == 0:
            return np.empty((0, 2), dtype=float)
        inner = self.inner_radius_m + float(margin)
        outer = self.outer_radius_m - float(margin)
        if outer <= inner:
            raise RuntimeError("El anillo no conserva espesor interior tras descontar el margen de interfaz.")
        radius = np.sqrt(rng.uniform(inner ** 2, outer ** 2, count))
        radial = (radius - self.inner_radius_m) / (self.outer_radius_m - self.inner_radius_m)
        angle = rng.uniform(0.0, 2.0 * math.pi, count) / (2.0 * math.pi)
        return self.local_to_domain(np.c_[radial, angle])

    def inside_bounds(self, bounds: tuple[float, float, float, float] | list[float]) -> bool:
        x0, x1, y0, y1 = bounds
        cx, cy = self.center
        r = self.outer_radius_m
        return cx - r >= x0 and cx + r <= x1 and cy - r >= y0 and cy + r <= y1

    def boundary_curves(self, n_per_curve: int) -> list[tuple[np.ndarray, np.ndarray, float]]:
        center = np.asarray(self.center, dtype=float)
        angle = 2.0 * np.pi * (np.arange(n_per_curve) + 0.5) / n_per_curve
        radial = np.c_[np.cos(angle), np.sin(angle)]
        return [
            (center + self.outer_radius_m * radial, radial, 2.0 * np.pi * self.outer_radius_m),
            (center + self.inner_radius_m * radial, -radial, 2.0 * np.pi * self.inner_radius_m),
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "annulus",
            "center": list(self.center),
            "inner_radius_m": float(self.inner_radius_m),
            "outer_radius_m": float(self.outer_radius_m),
        }


@dataclass(frozen=True)
class PolygonShape(Shape2D):
    """Región poligonal plana arbitraria definida por sus vértices ordenados.

    Permite representar zanjas trapezoidales, bancos de ductos escalonados o
    inclusiones geológicas irregulares manteniendo compatibilidad tanto en PINN
    como en generación de malla Gmsh/OpenCASCADE.
    """

    vertices: tuple[tuple[float, float], ...]
    shape_type: str = "polygon"

    def __post_init__(self) -> None:
        if len(self.vertices) < 3:
            raise ValueError("Un polígono requiere al menos 3 vértices.")
        if abs(self.signed_area()) <= 1e-16:
            raise ValueError("El polígono tiene área nula o sus vértices son colineales.")

    def signed_area(self) -> float:
        pts = np.asarray(self.vertices, dtype=float)
        x, y = pts[:, 0], pts[:, 1]
        return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

    def _bbox(self) -> tuple[float, float, float, float]:
        pts = np.asarray(self.vertices, dtype=float)
        return float(pts[:, 0].min()), float(pts[:, 0].max()), float(pts[:, 1].min()), float(pts[:, 1].max())

    def contains(self, x: np.ndarray, y: np.ndarray, tolerance: float = 0.0) -> np.ndarray:
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        shape = xa.shape
        px, py = xa.ravel(), ya.ravel()
        verts = np.asarray(self.vertices, dtype=float)
        inside = np.zeros(px.shape, dtype=bool)
        n = len(verts)
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            cross = ((y1 > py) != (y2 > py)) & (px < (x2 - x1) * (py - y1) / ((y2 - y1) + 1e-30) + x1)
            inside ^= cross
        if tolerance > 0.0:
            on_edge = self.boundary_distance(px, py) <= tolerance
            inside |= on_edge
        return inside.reshape(shape)

    def boundary_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        shape = xa.shape
        px, py = xa.ravel(), ya.ravel()
        verts = np.asarray(self.vertices, dtype=float)
        min_d2 = np.full(px.shape, np.inf, dtype=float)
        n = len(verts)
        for i in range(n):
            ax, ay = verts[i]
            bx, by = verts[(i + 1) % n]
            vx, vy = bx - ax, by - ay
            len2 = vx * vx + vy * vy
            if len2 <= 1e-30:
                d2 = (px - ax) ** 2 + (py - ay) ** 2
            else:
                t = np.clip(((px - ax) * vx + (py - ay) * vy) / len2, 0.0, 1.0)
                projx, projy = ax + t * vx, ay + t * vy
                d2 = (px - projx) ** 2 + (py - projy) ** 2
            min_d2 = np.minimum(min_d2, d2)
        return np.sqrt(min_d2).reshape(shape)

    def domain_to_local(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        x0, x1, y0, y1 = self._bbox()
        return np.c_[(xy[:, 0] - x0) / max(x1 - x0, 1e-12), (xy[:, 1] - y0) / max(y1 - y0, 1e-12)]

    def local_to_domain(self, local: np.ndarray) -> np.ndarray:
        local = np.asarray(local, dtype=float)
        x0, x1, y0, y1 = self._bbox()
        return np.c_[x0 + local[:, 0] * (x1 - x0), y0 + local[:, 1] * (y1 - y0)]

    def characteristic_size(self) -> float:
        x0, x1, y0, y1 = self._bbox()
        area = abs(self.signed_area())
        return float(min(x1 - x0, y1 - y0, math.sqrt(area)))

    def area(self) -> float:
        return float(abs(self.signed_area()))

    def interior_margin(self, span: float, factor: float = 1e-7) -> float:
        return min(factor * max(float(span), 1.0), 0.1 * self.characteristic_size())

    def sample_local(self, count: int, rng: np.random.Generator, margin: float = 0.0) -> np.ndarray:
        count = int(count)
        if count <= 0:
            return np.empty((0, 2), dtype=float)
        x0, x1, y0, y1 = self._bbox()
        accepted = []
        remaining = count
        for _ in range(1000):
            batch = max(128, 4 * remaining)
            cand_x = rng.uniform(x0, x1, batch)
            cand_y = rng.uniform(y0, y1, batch)
            mask = self.contains(cand_x, cand_y, tolerance=0.0) & (self.boundary_distance(cand_x, cand_y) > margin)
            pts = np.c_[cand_x[mask], cand_y[mask]]
            if len(pts):
                take = pts[:remaining]
                accepted.append(take)
                remaining -= len(take)
                if remaining == 0:
                    return np.vstack(accepted)
        raise RuntimeError("No se pudieron muestrear suficientes puntos interiores dentro del polígono.")

    def inside_bounds(self, bounds: tuple[float, float, float, float] | list[float]) -> bool:
        bx0, bx1, by0, by1 = bounds
        x0, x1, y0, y1 = self._bbox()
        return x0 >= bx0 and x1 <= bx1 and y0 >= by0 and y1 <= by1

    def boundary_curves(self, n_per_curve: int) -> list[tuple[np.ndarray, np.ndarray, float]]:
        verts = np.asarray(self.vertices, dtype=float)
        ccw = self.signed_area() > 0
        s = (np.arange(n_per_curve) + 0.5) / n_per_curve
        curves = []
        n = len(verts)
        for i in range(n):
            p0, p1 = verts[i], verts[(i + 1) % n]
            edge = p1 - p0
            length = float(np.linalg.norm(edge))
            if length <= 1e-15:
                continue
            # Normal exterior hacia afuera según orientación del polígono
            nx, ny = (edge[1] / length, -edge[0] / length) if ccw else (-edge[1] / length, edge[0] / length)
            xy = p0[None, :] + s[:, None] * edge[None, :]
            curves.append((xy, np.tile((nx, ny), (n_per_curve, 1)), length))
        return curves

    def to_dict(self) -> dict[str, Any]:
        return {"type": "polygon", "vertices": [list(v) for v in self.vertices]}


# ---------------------------------------------------------------------------
# Funciones utilitarias funcionales sobre diccionarios para retrocompatibilidad
# ---------------------------------------------------------------------------

def _as_shape(shape: dict[str, Any] | Shape2D) -> Shape2D:
    return shape if isinstance(shape, Shape2D) else Shape2D.from_dict(shape)


def contains(shape: dict[str, Any] | Shape2D, x: np.ndarray, y: np.ndarray, tolerance: float = 0.0) -> np.ndarray:
    return _as_shape(shape).contains(x, y, tolerance=tolerance)


def boundary_distance(shape: dict[str, Any] | Shape2D, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return _as_shape(shape).boundary_distance(x, y)


def domain_to_local(shape: dict[str, Any] | Shape2D, xy: np.ndarray) -> np.ndarray:
    return _as_shape(shape).domain_to_local(xy)


def local_to_domain(shape: dict[str, Any] | Shape2D, local: np.ndarray) -> np.ndarray:
    return _as_shape(shape).local_to_domain(local)


def interior_margin(shape: dict[str, Any] | Shape2D, span: float, factor: float = 1e-7) -> float:
    return _as_shape(shape).interior_margin(span, factor=factor)


def characteristic_size(shape: dict[str, Any] | Shape2D) -> float:
    return _as_shape(shape).characteristic_size()


def ring_offset(shape: dict[str, Any] | Shape2D) -> float:
    return _as_shape(shape).ring_offset()


def boundary_offset(shape: dict[str, Any] | Shape2D) -> float:
    return _as_shape(shape).boundary_offset()


def sample_local(shape: dict[str, Any] | Shape2D, count: int, rng: np.random.Generator, margin: float = 0.0) -> np.ndarray:
    return _as_shape(shape).sample_local(count, rng, margin=margin)


def inside_bounds(shape: dict[str, Any] | Shape2D, bounds: tuple[float, float, float, float] | list[float]) -> bool:
    return _as_shape(shape).inside_bounds(bounds)
