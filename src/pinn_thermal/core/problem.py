"""Definición unificada de un problema térmico 2D (``ThermalProblem2D``) y catálogo de casos.

Consolida en una sola clase orientada a objetos:
- Carga y validación desde archivo JSON o diccionario (``pinn-v4-problem-1`` / ``pinn-v3-problem-1``).
- Huella física determinista SHA-256 compartida entre los motores PINN y FEniCSx.
- Consultas de pertenencia por prioridad, muestreo interior geométrico/focalizado,
  segmentos de frontera externa y pares de anillos de interfaz.
- Coordenadas locales por región y características log-radiales para cables.

Importación **100 % libre de PyTorch**; las conversiones a tensor importan ``torch``
únicamente dentro de los métodos tensores correspondientes.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from pinn_thermal.core.fields import FieldSpec, evaluate_field_numpy, evaluate_field_torch
from pinn_thermal.core.schema import SCHEMA_V4, validate_problem
from pinn_thermal.core.shapes import Shape2D

NORMAL_OFFSET: float = 1e-7
MEMBERSHIP_TOLERANCE: float = 1e-8
INTERFACE_MARGIN_FACTOR: float = 1e-7
SCALE_GRID: tuple[int, int] = (401, 201)
MAX_SAMPLE_ATTEMPTS: int = 1000
SIDE_NORMALS: dict[str, tuple[float, float]] = {
    "bottom": (0.0, -1.0),
    "top": (0.0, 1.0),
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
}


@dataclass(frozen=True)
class RegionSpec:
    """Descriptor inmutable de una subregión material del dominio."""

    id: int
    name: str
    kind: str
    priority: int
    shape: Shape2D
    conductivity_spec: FieldSpec
    source_spec: FieldSpec
    heat_capacity_spec: FieldSpec | None = None
    axis: int = 0
    lower: float = 0.0
    upper: float = 1.0


@dataclass(frozen=True)
class CableLayer:
    """Capa concéntrica (conductor, aislamiento, pantalla o cubierta) de un cable."""

    cable_index: int
    region_id: int
    center: tuple[float, float]
    inner_radius: float
    outer_radius: float


@dataclass(frozen=True)
class CableGroup:
    """Agrupación geométrica de todas las capas concéntricas de un mismo cable."""

    index: int
    center: tuple[float, float]
    radius: float
    layers: tuple[CableLayer, ...]


@dataclass(frozen=True)
class CircularInterface:
    """Interfaz circular entre capas vecinas de un cable o entre cubierta y suelo."""

    name: str
    left: int
    right: int
    center: tuple[float, float]
    radius: float
    cable_index: int


@dataclass(frozen=True)
class InterfaceCloud:
    """Nube de colocación de una interfaz con anillos simétricos en ambos materiales."""

    name: str
    left: int
    right: int
    xy: np.ndarray
    normal: np.ndarray
    weight: float
    axis: int = -1
    xy_right: np.ndarray | None = None


@dataclass(frozen=True)
class BoundaryPiece:
    """Segmento muestreado de una frontera externa perteneciente a una sola región."""

    side: str
    region: int
    xy: np.ndarray
    normal: np.ndarray
    weight: float
    bc: dict[str, Any]
    length: float = 0.0


def canonical_physics_payload(doc: dict[str, Any]) -> dict[str, Any]:
    """Extrae exclusivamente las claves que determinan la física matemática del problema."""
    payload: dict[str, Any] = {
        "schema": SCHEMA_V4,
        "id": doc["id"],
        "domain": doc["domain"],
        "regions": doc["regions"],
        "boundaries": doc["boundaries"],
    }
    if "temperature_scale_K" in doc:
        payload["temperature_scale_K"] = doc["temperature_scale_K"]
    if "interfaces" in doc:
        payload["interfaces"] = doc["interfaces"]
    return payload


def compute_physics_sha256(doc: dict[str, Any]) -> str:
    """Calcula la huella SHA-256 determinista compartida entre PINN y FEniCSx."""
    encoded = json.dumps(canonical_physics_payload(doc), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ThermalProblem2D:
    """Representación unificada de un problema térmico 2D estacionario/transitorio."""

    reference_label: str = "referencia declarada en el archivo del problema"

    def __init__(self, doc: dict[str, Any]) -> None:
        validated = validate_problem(doc)
        self.doc: dict[str, Any] = validated
        self.id: str = validated["id"]
        self.name: str = validated["id"]
        self.title: str = str(validated.get("title", self.id))
        self.bounds: tuple[float, float, float, float] = tuple(float(v) for v in validated["domain"]["bounds"])  # type: ignore[assignment]
        self.domain_shape: str = validated["domain"]["shape"]
        self.physics_sha256: str = compute_physics_sha256(validated)
        self.family: str = self._infer_family(validated)

        self.regions: tuple[RegionSpec, ...] = tuple(
            self._build_region(idx, entry) for idx, entry in enumerate(validated["regions"])
        )
        self.region_by_name: dict[str, RegionSpec] = {r.name: r for r in self.regions}
        self.cable_layout: tuple[CableGroup, ...] = self._build_cable_layout()
        self.circular_interfaces: tuple[CircularInterface, ...] = self._build_circular_interfaces()
        self.source_disks: tuple[Shape2D, ...] = tuple(
            r.shape
            for r in self.regions
            if r.shape.shape_type == "disk"
            and bool(
                np.any(
                    r.source_spec.evaluate_numpy(
                        np.array([r.shape.center[0]]), np.array([r.shape.center[1]])  # type: ignore[attr-defined]
                    )
                    != 0.0
                )
            )
        )

        self.T_ref: float = self._reference_temperature()
        self.T_scale: float = self._temperature_scale()
        self.T_level: float = self.T_ref

        # Pre-cálculo de escalas térmicas adimensionales por región
        grid_xy, cell_area = self._scale_grid()
        owner = self.owner_index(grid_xy)
        exact_areas = self._exact_cable_areas()
        local_rng = np.random.default_rng(20261125)
        self.k_ref: dict[int, float] = {}
        self.region_power: dict[int, float] = {}
        self.source_abs: dict[int, float] = {}
        self.areas: dict[int, float] = {}
        absolute = 0.0
        for region in self.regions:
            mine = owner == region.id
            hits = int(np.count_nonzero(mine))
            if region.id in exact_areas:
                self.areas[region.id] = exact_areas[region.id]
                points = grid_xy[mine] if hits >= 64 else self._representative_point(region).reshape(1, 2)
                if hits < 64:
                    local_pts = region.shape.sample_local(4096, local_rng)
                    owned = self.owner(local_pts[:, 0], local_pts[:, 1]) == region.id
                    if np.any(owned):
                        points = local_pts[owned]
            elif hits >= 64:
                self.areas[region.id] = max(float(hits) * cell_area, 1e-12)
                points = grid_xy[mine]
            else:
                local_pts = region.shape.sample_local(4096, local_rng)
                owned = self.owner(local_pts[:, 0], local_pts[:, 1]) == region.id
                frac = float(np.mean(owned)) if len(owned) else 0.0
                self.areas[region.id] = max(region.shape.area() * frac, 1e-12)
                points = local_pts[owned] if np.any(owned) else self._representative_point(region).reshape(1, 2)
            k_values = region.conductivity_spec.evaluate_numpy(points[:, 0], points[:, 1])
            q_values = region.source_spec.evaluate_numpy(points[:, 0], points[:, 1])
            self.k_ref[region.id] = float(np.mean(k_values))
            self.region_power[region.id] = float(np.mean(q_values) * self.areas[region.id])
            self.source_abs[region.id] = float(np.mean(np.abs(q_values)))
            absolute += self.source_abs[region.id] * self.areas[region.id]

        characteristic = max(self.k_ref.values()) * self.T_scale
        self._source_power_abs: float = absolute
        self.power_total: float = sum(self.region_power.values())
        self.power_scale: float = absolute if absolute > 0.0 else characteristic

    @classmethod
    def from_json(cls, source: str | Path | dict[str, Any]) -> ThermalProblem2D:
        """Carga y construye ``ThermalProblem2D`` desde un archivo JSON o diccionario."""
        if isinstance(source, ThermalProblem2D):
            return source
        if isinstance(source, dict):
            return cls(source)
        path = Path(source)
        doc = json.loads(path.read_text(encoding="utf-8-sig"))
        return cls(doc)

    # ------------------------------------------------------------------
    # Construcción interna de regiones y cables
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_family(doc: dict[str, Any]) -> str:
        declared = doc.get("family")
        if declared:
            return str(declared)
        if doc.get("domain", {}).get("shape") == "annulus":
            return "annulus"
        if doc.get("reference", {}).get("exact") is not None:
            return "mms"
        for region in doc["regions"]:
            if region.get("kind") == "conductor":
                return "cable"
            shape = region["shape"]
            if shape["type"] == "disk":
                source = region.get("source", 0.0)
                pt = np.asarray(shape["center"], dtype=float).reshape(1, 2)
                if np.any(evaluate_field_numpy(source, pt[:, 0], pt[:, 1]) != 0.0):
                    return "cable"
        return "generic"

    def _build_region(self, index: int, entry: dict[str, Any]) -> RegionSpec:
        shape = Shape2D.from_dict(entry["shape"])
        if shape.shape_type == "rectangle":
            lower, upper = float(shape.bounds[0]), float(shape.bounds[1])  # type: ignore[attr-defined]
        else:
            lower, upper = 0.0, 1.0
        declared_kind = entry.get("kind")
        if declared_kind is not None:
            kind = str(declared_kind)
        elif self.family == "cable" and shape.shape_type in ("disk", "annulus"):
            kind = "conductor"
        else:
            kind = "soil"
        return RegionSpec(
            id=index,
            name=entry["id"],
            kind=kind,
            priority=int(entry["priority"]),
            shape=shape,
            conductivity_spec=FieldSpec.from_dict(entry["conductivity"], f"region {entry['id']!r}.conductivity"),
            source_spec=FieldSpec.from_dict(entry.get("source", 0.0), f"region {entry['id']!r}.source"),
            heat_capacity_spec=(
                FieldSpec.from_dict(entry["heat_capacity"], f"region {entry['id']!r}.heat_capacity")
                if "heat_capacity" in entry
                else None
            ),
            axis=0,
            lower=lower,
            upper=upper,
        )

    def _build_cable_layout(self) -> tuple[CableGroup, ...]:
        if self.family != "cable":
            return ()
        grouped: dict[tuple[float, float], list[tuple[float, float, int]]] = {}
        for region in self.regions:
            if region.shape.shape_type == "disk":
                center = region.shape.center  # type: ignore[attr-defined]
                inner, outer = 0.0, float(region.shape.radius_m)  # type: ignore[attr-defined]
            elif region.shape.shape_type == "annulus":
                center = region.shape.center  # type: ignore[attr-defined]
                inner = float(region.shape.inner_radius_m)  # type: ignore[attr-defined]
                outer = float(region.shape.outer_radius_m)  # type: ignore[attr-defined]
            else:
                continue
            key = (float(center[0]), float(center[1]))
            grouped.setdefault(key, []).append((inner, outer, region.id))
        cables: list[CableGroup] = []
        for cable_index, (center, entries) in enumerate(sorted(grouped.items())):
            entries.sort(key=lambda item: (item[0], item[1], item[2]))
            layers = tuple(
                CableLayer(cable_index, region_id, center, inner, outer)
                for inner, outer, region_id in entries
            )
            cables.append(CableGroup(cable_index, center, max(l.outer_radius for l in layers), layers))
        return tuple(cables)

    @property
    def supports_exact_cable_interfaces(self) -> bool:
        """Indica si el problema admite trazas de Fourier circulares exactas (``M1_EXACT``)."""
        if self.family != "cable" or not self.cable_layout or self.domain_shape != "rectangle":
            return False
        shell_regions = {layer.region_id for cable in self.cable_layout for layer in cable.layers}
        full_bounds = list(self.bounds)
        backgrounds = [
            r.id
            for r in self.regions
            if r.shape.shape_type == "rectangle"
            and list(r.shape.bounds) == full_bounds  # type: ignore[attr-defined]
            and r.id not in shell_regions
        ]
        return len(backgrounds) == 1 and len(shell_regions) + len(backgrounds) == len(self.regions)

    def _exact_cable_areas(self) -> dict[int, float]:
        if self.family != "cable" or not self.cable_layout or self.domain_shape != "rectangle":
            return {}
        layer_ids = {layer.region_id for cable in self.cable_layout for layer in cable.layers}
        backgrounds = [r.id for r in self.regions if r.id not in layer_ids]
        if len(backgrounds) != 1:
            return {}
        bg_id = backgrounds[0]
        bg_shape = self.regions[bg_id].shape
        if bg_shape.shape_type != "rectangle" or list(bg_shape.bounds) != list(self.bounds):  # type: ignore[attr-defined]
            return {}
        x0, x1, y0, y1 = self.bounds
        for idx, cable in enumerate(self.cable_layout):
            cx, cy = cable.center
            if min(cx - x0, x1 - cx, cy - y0, y1 - cy) < cable.radius:
                return {}
            for other in self.cable_layout[idx + 1:]:
                if math.hypot(cx - other.center[0], cy - other.center[1]) < cable.radius + other.radius:
                    return {}
        areas: dict[int, float] = {}
        for cable in self.cable_layout:
            for layer in cable.layers:
                areas[layer.region_id] = math.pi * (layer.outer_radius ** 2 - layer.inner_radius ** 2)
        domain_area = (x1 - x0) * (y1 - y0)
        areas[bg_id] = domain_area - sum(math.pi * cable.radius ** 2 for cable in self.cable_layout)
        return areas

    def _representative_point(self, region: RegionSpec) -> np.ndarray:
        shape = region.shape
        if shape.shape_type == "rectangle":
            x0, x1, y0, y1 = shape.bounds  # type: ignore[attr-defined]
            return np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)], dtype=float)
        if shape.shape_type == "disk":
            return np.asarray(shape.center, dtype=float)  # type: ignore[attr-defined]
        if shape.shape_type == "annulus":
            r = math.sqrt(0.5 * (shape.inner_radius_m ** 2 + shape.outer_radius_m ** 2))  # type: ignore[attr-defined]
            return np.asarray(shape.center, dtype=float) + np.array([r, 0.0])  # type: ignore[attr-defined]
        verts = np.asarray(shape.vertices, dtype=float)  # type: ignore[attr-defined]
        return verts.mean(axis=0)

    def _build_circular_interfaces(self) -> tuple[CircularInterface, ...]:
        if not self.cable_layout:
            return ()
        span = max(self.bounds[1] - self.bounds[0], self.bounds[3] - self.bounds[2], 1.0)
        probe_dist = max(NORMAL_OFFSET, 2.0 * MEMBERSHIP_TOLERANCE * span)
        interfaces: list[CircularInterface] = []
        for cable in self.cable_layout:
            for layer_index, layer in enumerate(cable.layers):
                right = (
                    cable.layers[layer_index + 1].region_id
                    if layer_index + 1 < len(cable.layers)
                    else None
                )
                if right is None:
                    probe = np.asarray(cable.center, dtype=float) + np.array([layer.outer_radius + probe_dist, 0.0])
                    right = int(self.owner_index(probe.reshape(1, 2))[0])
                if right < 0 or (self.regions[right].kind == "conductor" and right == layer.region_id):
                    raise ValueError(f"La capa de cable {layer.region_id!r} carece de región vecina exterior.")
                interfaces.append(
                    CircularInterface(
                        name=f"cable_{cable.index}_radius_{layer.outer_radius:g}",
                        left=layer.region_id,
                        right=right,
                        center=cable.center,
                        radius=layer.outer_radius,
                        cable_index=cable.index,
                    )
                )
        return tuple(interfaces)

    # ------------------------------------------------------------------
    # Referencias térmicas y lados del dominio
    # ------------------------------------------------------------------

    def _scale_grid(self) -> tuple[np.ndarray, float]:
        x0, x1, y0, y1 = self.bounds
        gx, gy = np.meshgrid(np.linspace(x0, x1, SCALE_GRID[0]), np.linspace(y0, y1, SCALE_GRID[1]))
        area = (x1 - x0) * (y1 - y0) / (SCALE_GRID[0] * SCALE_GRID[1])
        return np.c_[gx.ravel(), gy.ravel()], float(area)

    def _domain_sides(self, n: int) -> dict[str, dict[str, Any]]:
        if self.domain_shape == "annulus":
            domain = self.doc["domain"]
            center = np.asarray(domain["center"], dtype=float)
            angle = (np.arange(n) + 0.5) * 2.0 * np.pi / n
            circle = np.c_[np.cos(angle), np.sin(angle)]
            ri, ro = float(domain["inner_radius_m"]), float(domain["outer_radius_m"])
            return {
                "inner": {"xy": center + ri * circle, "normal": -circle, "length": 2.0 * np.pi * ri},
                "outer": {"xy": center + ro * circle, "normal": circle, "length": 2.0 * np.pi * ro},
            }
        x0, x1, y0, y1 = self.bounds
        s = (np.arange(n) + 0.5) / n
        ones = np.ones(n)
        return {
            "bottom": {"xy": np.c_[x0 + (x1 - x0) * s, y0 * ones], "normal": np.tile(SIDE_NORMALS["bottom"], (n, 1)), "length": x1 - x0},
            "top": {"xy": np.c_[x0 + (x1 - x0) * s, y1 * ones], "normal": np.tile(SIDE_NORMALS["top"], (n, 1)), "length": x1 - x0},
            "left": {"xy": np.c_[x0 * ones, y0 + (y1 - y0) * s], "normal": np.tile(SIDE_NORMALS["left"], (n, 1)), "length": y1 - y0},
            "right": {"xy": np.c_[x1 * ones, y0 + (y1 - y0) * s], "normal": np.tile(SIDE_NORMALS["right"], (n, 1)), "length": y1 - y0},
        }

    def _reference_on_side(self, side: str) -> np.ndarray | None:
        cond = self.doc["boundaries"][side]
        spec = cond.get("value") if cond["type"] == "dirichlet" else cond.get("T_inf_C")
        if spec is None:
            return None
        xy = self._domain_sides(33)[side]["xy"]
        return evaluate_field_numpy(spec, xy[:, 0], xy[:, 1])

    def _reference_temperature(self) -> float:
        levels = [v for v in (self._reference_on_side(s) for s in self.doc["boundaries"]) if v is not None]
        if not levels:
            return 20.0
        return float(min(np.min(v) for v in levels))

    def _temperature_scale(self) -> float:
        if "temperature_scale_K" in self.doc:
            return float(self.doc["temperature_scale_K"])
        levels = [v for v in (self._reference_on_side(s) for s in self.doc["boundaries"]) if v is not None]
        if not levels:
            return 50.0
        return float(max(1.0, max(np.max(v) - self.T_ref for v in levels)))

    # ------------------------------------------------------------------
    # Superficie geométrica y evaluación NumPy (torch-free)
    # ------------------------------------------------------------------

    def owner(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Devuelve el índice de región propietaria de cada punto ``(x, y)`` según ``priority``."""
        xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        index = np.full(xa.shape, -1, dtype=int)
        x0, x1, y0, y1 = self.bounds
        tol = MEMBERSHIP_TOLERANCE * max(x1 - x0, y1 - y0, 1.0)
        for region in sorted(self.regions, key=lambda item: item.priority):
            index[region.shape.contains(xa, ya, tolerance=tol)] = region.id
        return index

    def owner_index(self, xy: np.ndarray) -> np.ndarray:
        """Índice de región propietaria para una matriz ``xy`` de forma ``(N, 2)``."""
        pts = np.asarray(xy, dtype=float)
        return self.owner(pts[:, 0], pts[:, 1])

    def contains_points(self, region: RegionSpec | int, points: np.ndarray) -> np.ndarray:
        rid = region.id if isinstance(region, RegionSpec) else int(region)
        return self.owner_index(points) == rid

    def conductors(self) -> list[int]:
        """Lista de identificadores numéricos de las regiones conductoras."""
        return [r.id for r in self.regions if r.kind == "conductor"]

    def area(self, region: RegionSpec | int) -> float:
        rid = region.id if isinstance(region, RegionSpec) else int(region)
        return self.areas[rid]

    def domain_to_local(self, region: RegionSpec | int, xy: np.ndarray) -> np.ndarray:
        reg = region if isinstance(region, RegionSpec) else self.regions[int(region)]
        return reg.shape.domain_to_local(xy)

    def local_to_domain(self, region: RegionSpec | int, local: np.ndarray) -> np.ndarray:
        reg = region if isinstance(region, RegionSpec) else self.regions[int(region)]
        return reg.shape.local_to_domain(local)

    def field_numpy(self, name: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Evalúa ``conductivity``, ``source`` o ``heat_capacity`` en NumPy desde la región propietaria."""
        xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        owners = self.owner(xa, ya)
        out = np.full(xa.shape, np.nan, dtype=float)
        for region in self.regions:
            mine = owners == region.id
            if not np.any(mine):
                continue
            if name == "conductivity":
                spec = region.conductivity_spec
            elif name == "source":
                spec = region.source_spec
            elif name == "heat_capacity":
                if region.heat_capacity_spec is None:
                    raise KeyError(f"La región {region.name!r} no declara heat_capacity.")
                spec = region.heat_capacity_spec
            else:
                raise KeyError(f"Campo físico desconocido: {name!r}.")
            out[mine] = spec.evaluate_numpy(xa[mine], ya[mine])
        return out

    def reference(self, xy: np.ndarray, labels: Any = None) -> np.ndarray | None:
        """Evalúa ``reference.exact`` si el JSON declara solución exacta."""
        spec = self.doc.get("reference", {}).get("exact")
        if spec is None:
            return None
        pts = np.asarray(xy, dtype=float)
        return evaluate_field_numpy(spec, pts[:, 0], pts[:, 1])

    def _clear_membership(self, region: RegionSpec, points: np.ndarray) -> np.ndarray:
        owners = self.owner_index(points)
        span = max(self.bounds[1] - self.bounds[0], self.bounds[3] - self.bounds[2], 1.0)
        margin = region.shape.interior_margin(span, INTERFACE_MARGIN_FACTOR)
        dist = region.shape.boundary_distance(points[:, 0], points[:, 1])
        return (owners == region.id) & (dist > margin)

    def sample_region_interior(
        self,
        region: RegionSpec | int,
        count: int,
        rng: np.random.Generator | int = 0,
    ) -> np.ndarray:
        """Muestrea puntos interiores uniformes en área pertenecientes exclusivamente a ``region``."""
        reg = region if isinstance(region, RegionSpec) else self.regions[int(region)]
        gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(int(rng))
        span = max(self.bounds[1] - self.bounds[0], self.bounds[3] - self.bounds[2], 1.0)
        margin = reg.shape.interior_margin(span, INTERFACE_MARGIN_FACTOR)
        accepted: list[np.ndarray] = []
        remaining = int(count)
        for _ in range(MAX_SAMPLE_ATTEMPTS):
            cand = reg.shape.sample_local(max(64, 2 * remaining), gen, margin)
            owned = cand[self.owner(cand[:, 0], cand[:, 1]) == reg.id]
            if len(owned):
                take = owned[:remaining]
                accepted.append(take)
                remaining -= len(take)
                if remaining == 0:
                    return np.vstack(accepted)
        raise RuntimeError(
            f"La región {reg.name!r} posee muy poca área efectiva no solapada para muestrear {count} puntos."
        )

    def sample_interior(self, region: RegionSpec | int, n: int, seed: int = 0, policy: str = "uniform") -> np.ndarray:
        if policy == "paired":
            raise NotImplementedError("La política 'paired' heredada no aplica a documentos de regiones.")
        return self.sample_region_interior(region, n, seed)

    def focused_interior_pool(self, region: RegionSpec, n: int, seed: int, policy: str = "mixed") -> np.ndarray:
        """Combina mitad uniforme y mitad concentrada log-radialmente cerca de fuentes circulares."""
        uniform_count = n // 2
        uniform = self.sample_interior(region, uniform_count, seed, policy)
        if not self.source_disks or region.shape.shape_type != "rectangle":
            return np.vstack([uniform, self.sample_interior(region, n - uniform_count, seed + 1, policy)])
        rng = np.random.default_rng(seed + 2)
        x0, x1, y0, y1 = self.bounds
        r_max = max(min(x1 - x0, y1 - y0) * 0.5, 4.0 * max(d.radius_m for d in self.source_disks))  # type: ignore[attr-defined]
        focused: list[np.ndarray] = []
        accepted = 0
        target = n - uniform_count
        for _ in range(MAX_SAMPLE_ATTEMPTS):
            count = max(256, target * 2)
            idxs = rng.integers(0, len(self.source_disks), count)
            radii = np.array([self.source_disks[i].radius_m for i in idxs], dtype=float)  # type: ignore[attr-defined]
            radii *= np.exp(rng.uniform(1e-6, np.log(r_max / radii)))
            angles = rng.uniform(0.0, 2.0 * np.pi, count)
            centers_x = np.array([self.source_disks[i].center[0] for i in idxs], dtype=float)  # type: ignore[attr-defined]
            centers_y = np.array([self.source_disks[i].center[1] for i in idxs], dtype=float)  # type: ignore[attr-defined]
            cand = np.c_[centers_x + radii * np.cos(angles), centers_y + radii * np.sin(angles)]
            mine = cand[self._clear_membership(region, cand)]
            if len(mine):
                focused.append(mine)
                accepted += len(mine)
                if accepted >= target:
                    return np.vstack([uniform, np.vstack(focused)[:target]])
        return np.vstack([uniform, self.sample_interior(region, target, seed + 3, policy)])

    def _boundary_rows(self, region_id: int, xy: np.ndarray, normal: np.ndarray, rows: int = 2) -> tuple[np.ndarray, np.ndarray]:
        if rows == 1:
            return xy, normal
        offset = self.regions[region_id].shape.boundary_offset()
        inner = xy - offset * normal
        keep = self.owner(inner[:, 0], inner[:, 1]) == region_id
        return np.vstack([xy, inner[keep]]), np.vstack([normal, normal[keep]])

    def sample_boundary_regions(self, n_per_piece: int, rows: int = 2) -> list[dict[str, Any]]:
        """Muestrea cada segmento de frontera externa perteneciente a una sola región."""
        if rows not in (1, 2):
            raise ValueError("rows debe ser 1 o 2.")
        if n_per_piece < 1:
            raise ValueError("n_per_piece debe ser positivo.")
        if self.domain_shape == "annulus":
            domain = self.doc["domain"]
            center = np.asarray(domain["center"], dtype=float)
            result: list[dict[str, Any]] = []
            for side, radius, sign in (
                ("inner", float(domain["inner_radius_m"]), -1.0),
                ("outer", float(domain["outer_radius_m"]), 1.0),
            ):
                angle = 2.0 * np.pi * (np.arange(n_per_piece) + 0.5) / n_per_piece
                dirs = np.c_[np.cos(angle), np.sin(angle)]
                xy = center + radius * dirs
                normals = sign * dirs
                owners = self.owner(xy[:, 0], xy[:, 1])
                for region_id in np.unique(owners[owners >= 0]):
                    sel = owners == region_id
                    r_xy, r_n = self._boundary_rows(int(region_id), xy[sel], normals[sel], rows)
                    result.append(
                        {
                            "side": side,
                            "region": int(region_id),
                            "xy": r_xy,
                            "normal": r_n,
                            "length": 2.0 * np.pi * radius * float(np.count_nonzero(sel)) / n_per_piece,
                            "condition": self.doc["boundaries"][side],
                        }
                    )
            return result

        x0, x1, y0, y1 = self.bounds
        grouped: dict[str, list[dict[str, Any]]] = {}
        side_geom = {
            "bottom": (1, y0, x0, x1),
            "top": (1, y1, x0, x1),
            "left": (0, x0, y0, y1),
            "right": (0, x1, y0, y1),
        }
        for side, (fixed_axis, fixed, tan0, tan1) in side_geom.items():
            cuts = {float(tan0), float(tan1)}
            for region in self.regions:
                s = region.shape
                if s.shape_type == "rectangle":
                    a0, a1, b0, b1 = s.bounds  # type: ignore[attr-defined]
                    touches = (
                        np.isclose(fixed, a0 if side == "left" else a1)
                        if fixed_axis == 0
                        else np.isclose(fixed, b0 if side == "bottom" else b1)
                    )
                    if touches:
                        cuts.update((max(tan0, b0 if fixed_axis == 0 else a0), min(tan1, b1 if fixed_axis == 0 else a1)))
                elif s.shape_type in ("disk", "annulus"):
                    center = s.center  # type: ignore[attr-defined]
                    offset = fixed - center[fixed_axis]
                    tan_c = center[1 - fixed_axis]
                    radii = [s.radius_m] if s.shape_type == "disk" else [s.inner_radius_m, s.outer_radius_m]  # type: ignore[attr-defined]
                    for r in radii:
                        if abs(offset) <= r:
                            half = math.sqrt(max(r * r - offset * offset, 0.0))
                            cuts.update((max(tan0, tan_c - half), min(tan1, tan_c + half)))
            ordered = sorted(v for v in cuts if tan0 <= v <= tan1)
            for lo, hi in zip(ordered, ordered[1:]):
                if hi - lo <= np.finfo(float).eps * max(abs(lo), abs(hi), 1.0):
                    continue
                mid = 0.5 * (lo + hi)
                probe = np.array([[fixed, mid]]) if fixed_axis == 0 else np.array([[mid, fixed]])
                region_id = int(self.owner(probe[:, 0], probe[:, 1])[0])
                if region_id < 0:
                    continue
                tangent = lo + (np.arange(n_per_piece) + 0.5) * (hi - lo) / n_per_piece
                xy = np.c_[np.full(n_per_piece, fixed), tangent] if fixed_axis == 0 else np.c_[tangent, np.full(n_per_piece, fixed)]
                r_xy, r_n = self._boundary_rows(region_id, xy, np.tile(SIDE_NORMALS[side], (n_per_piece, 1)), rows)
                grouped.setdefault(side, []).append(
                    {
                        "side": side,
                        "region": region_id,
                        "xy": r_xy,
                        "normal": r_n,
                        "length": float(hi - lo),
                        "condition": self.doc["boundaries"][side],
                    }
                )
        return [piece for pieces in grouped.values() for piece in pieces]

    def sample_interfaces(self, n_per_curve: int) -> list[dict[str, Any]]:
        """Genera pares de anillos simétricos a ambos lados de cada interfaz material."""
        span = max(self.bounds[1] - self.bounds[0], self.bounds[3] - self.bounds[2], 1.0)
        tol = MEMBERSHIP_TOLERANCE * span
        probe = max(1e-7, 2.0 * tol)
        result: list[dict[str, Any]] = []
        for region in self.regions:
            for c_idx, (xy, normal, length) in enumerate(region.shape.boundary_curves(n_per_curve)):
                outside = self.owner(*(xy + probe * normal).T)
                for neighbor in np.unique(outside[(outside >= 0) & (outside != region.id)]):
                    neighbor_id = int(neighbor)
                    on_neighbor = self.regions[neighbor_id].shape.boundary_distance(xy[:, 0], xy[:, 1]) <= tol
                    selected = (outside == neighbor_id) & (~on_neighbor | (region.id < neighbor_id))
                    delta = min(region.shape.ring_offset(), self.regions[neighbor_id].shape.ring_offset())
                    left_ring = xy - delta * normal
                    right_ring = xy + delta * normal
                    keep = (
                        selected
                        & (self.owner(left_ring[:, 0], left_ring[:, 1]) == region.id)
                        & (self.owner(right_ring[:, 0], right_ring[:, 1]) == neighbor_id)
                    )
                    if not np.any(keep):
                        continue
                    result.append(
                        {
                            "name": f"{region.name}->{self.regions[neighbor_id].name}@{c_idx}",
                            "left": region.id,
                            "right": neighbor_id,
                            "xy": left_ring[keep],
                            "xy_right": right_ring[keep],
                            "normal": normal[keep],
                            "weight": float(length) / n_per_curve,
                        }
                    )
        # Interfaces planas internas declaradas explícitamente sobre una misma región
        for declared in self.doc.get("interfaces", ()):
            left_name, right_name = declared["regions"]
            if left_name != right_name:
                continue
            reg = self.region_by_name[left_name]
            if reg.shape.shape_type != "rectangle":
                raise ValueError(f"La interfaz declarada {declared['id']!r} requiere región rectangular.")
            x0, x1, y0, y1 = reg.shape.bounds  # type: ignore[attr-defined]
            axis = 0 if declared["axis"] == "x" else 1
            lo, hi = (y0, y1) if axis == 0 else (x0, x1)
            tangent = lo + (hi - lo) * (np.arange(n_per_curve) + 0.5) / n_per_curve
            pos = np.full(n_per_curve, float(declared["position"]))
            xy = np.c_[pos, tangent] if axis == 0 else np.c_[tangent, pos]
            normal = np.tile([1.0, 0.0] if axis == 0 else [0.0, 1.0], (n_per_curve, 1))
            delta = reg.shape.ring_offset()
            l_ring, r_ring = xy - delta * normal, xy + delta * normal
            keep = (
                (self.owner(l_ring[:, 0], l_ring[:, 1]) == reg.id)
                & (self.owner(r_ring[:, 0], r_ring[:, 1]) == reg.id)
            )
            if not np.any(keep):
                raise ValueError(f"La interfaz declarada {declared['id']!r} cae fuera de su región.")
            result.append(
                {
                    "name": f"{declared['id']}@declared",
                    "left": reg.id,
                    "right": reg.id,
                    "xy": l_ring[keep],
                    "xy_right": r_ring[keep],
                    "normal": normal[keep],
                    "weight": float(hi - lo) / n_per_curve,
                }
            )
        return result

    def faces(self, n: int) -> list[InterfaceCloud]:
        """Interfaces adaptadas para el cálculo de pérdidas y auditoría."""
        return [
            InterfaceCloud(
                name=item["name"],
                left=item["left"],
                right=item["right"],
                xy=item["xy"],
                normal=item["normal"],
                weight=item["weight"],
                xy_right=item["xy_right"],
            )
            for item in self.sample_interfaces(n)
        ]

    def pieces(self, n: int) -> list[BoundaryPiece]:
        """Segmentos de frontera externa adaptados con sus condiciones límite."""
        out: list[BoundaryPiece] = []
        for item in self.sample_boundary_regions(n):
            count = len(item["xy"])
            if count:
                out.append(
                    BoundaryPiece(
                        side=item["side"],
                        region=item["region"],
                        xy=item["xy"],
                        normal=item["normal"],
                        weight=float(item["length"]) / count,
                        bc=self.boundary_condition(item["condition"]),
                        length=float(item["length"]),
                    )
                )
        return out

    def evaluation(self, per_region: int = 1000, seed: int = 20261024) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Nube determinista de evaluación con pesos de cuadratura por área."""
        xy_blocks, labels, weights = [], [], []
        for region in self.regions:
            pts = self.sample_interior(region, per_region, seed + region.id)
            xy_blocks.append(pts)
            labels.append(np.full(per_region, region.id, dtype=np.int64))
            weights.append(np.full(per_region, self.areas[region.id] / per_region, dtype=float))
        return np.vstack(xy_blocks), np.concatenate(labels), np.concatenate(weights)

    # ------------------------------------------------------------------
    # Superficie tensor PyTorch (importación diferida de torch)
    # ------------------------------------------------------------------

    def boundary_condition(self, condition: dict[str, Any]) -> dict[str, Any]:
        """Devuelve un descriptor de frontera con evaluadores tensores diferidos."""
        if condition["type"] == "dirichlet":
            return {"type": "dirichlet", "value": lambda xy, spec=condition["value"]: evaluate_field_torch(spec, xy[:, 0:1], xy[:, 1:2])}
        if condition["type"] == "neumann":
            return {"type": "neumann", "value": lambda xy, spec=condition["q_n_W_m2"]: evaluate_field_torch(spec, xy[:, 0:1], xy[:, 1:2])}
        return {
            "type": "robin",
            "h": float(condition["h_W_m2K"]),
            "T_inf": lambda xy, spec=condition["T_inf_C"]: evaluate_field_torch(spec, xy[:, 0:1], xy[:, 1:2]),
        }

    def conductivity(self, region: RegionSpec, xy: Any) -> Any:
        return evaluate_field_torch(region.conductivity_spec, xy[:, 0:1], xy[:, 1:2])

    def source(self, region: RegionSpec, xy: Any) -> Any:
        return evaluate_field_torch(region.source_spec, xy[:, 0:1], xy[:, 1:2])

    def heat_capacity(self, region: RegionSpec, xy: Any) -> Any:
        if region.heat_capacity_spec is None:
            raise ValueError(f"La región {region.name!r} no declara heat_capacity.")
        return evaluate_field_torch(region.heat_capacity_spec, xy[:, 0:1], xy[:, 1:2])

    def scales(self, region: RegionSpec) -> dict[str, float]:
        """Escalas físicas adimensionales propias de cada región."""
        k = self.k_ref[region.id]
        length = max(min(math.sqrt(self.areas[region.id]), region.shape.characteristic_size()), 1e-12)
        src = self.source_abs[region.id]
        if src > 0.0:
            local = src * (length ** 2) / k
        elif self._source_power_abs > 0.0:
            fourier_local = 2.0 * math.pi * self._source_power_abs * (length ** 2) / max(k * self.areas[region.id], 1e-12)
            local = min(self.T_scale, max(fourier_local, 1e-6 * self.T_scale))
        else:
            local = self.T_scale
        return {
            "temperature": self.T_scale,
            "local_temperature": local,
            "flux": k * local / length,
            "divergence": k * local / (length ** 2),
        }

    def amplitude(self, region: RegionSpec, partition: str) -> float:
        if self.source_abs[region.id] > 0.0 or region.kind != "soil":
            return self.scales(region)["local_temperature"]
        return self.T_scale

    def interface_flux_scale(self, face: InterfaceCloud) -> float:
        length = max(face.weight * len(face.xy), 1e-12)
        power = max(abs(self.region_power[face.left]), abs(self.region_power[face.right]))
        if power > 0.0:
            return max(power / length, 1e-12)
        return max(self.scales(self.regions[face.left])["flux"], self.scales(self.regions[face.right])["flux"], 1e-12)

    def input_size(self, region: RegionSpec, partition: str) -> int:
        stype = region.shape.shape_type
        local_size = 3 if stype == "annulus" else 2
        if partition == "global":
            return 2 + 3 + len(self.regions) + len(self.source_disks)
        return local_size + (len(self.source_disks) if stype == "rectangle" else 0)

    def normalized_xy(self, xy: Any) -> Any:
        import torch

        x0, x1, y0, y1 = self.bounds
        return torch.cat(
            [2.0 * (xy[:, 0:1] - x0) / (x1 - x0) - 1.0, 2.0 * (xy[:, 1:2] - y0) / (y1 - y0) - 1.0],
            dim=1,
        )

    def _local_tensor_coordinates(self, region: RegionSpec, xy: Any) -> Any:
        import torch

        s = region.shape
        if s.shape_type == "rectangle":
            x0, x1, y0, y1 = s.bounds  # type: ignore[attr-defined]
            return torch.cat(
                [2.0 * (xy[:, 0:1] - x0) / (x1 - x0) - 1.0, 2.0 * (xy[:, 1:2] - y0) / (y1 - y0) - 1.0],
                dim=1,
            )
        if s.shape_type == "polygon":
            x0, x1, y0, y1 = s._bbox()  # type: ignore[attr-defined]
            return torch.cat(
                [
                    2.0 * (xy[:, 0:1] - x0) / max(x1 - x0, 1e-12) - 1.0,
                    2.0 * (xy[:, 1:2] - y0) / max(y1 - y0, 1e-12) - 1.0,
                ],
                dim=1,
            )
        center = xy.new_tensor(s.center)  # type: ignore[attr-defined]
        delta = xy[:, :2] - center
        if s.shape_type == "disk":
            return delta / float(s.radius_m)  # type: ignore[attr-defined]
        radius = torch.linalg.vector_norm(delta, dim=1, keepdim=True).clamp_min(torch.finfo(xy.dtype).tiny)
        ri, ro = float(s.inner_radius_m), float(s.outer_radius_m)  # type: ignore[attr-defined]
        radial = 2.0 * (radius - ri) / (ro - ri) - 1.0
        return torch.cat([radial, delta / radius], dim=1)

    def _source_distance_features(self, xy: Any, bounds: tuple[float, float, float, float]) -> list[Any]:
        import torch

        x0, x1, y0, y1 = bounds
        corners = xy.new_tensor([[x0, y0], [x0, y1], [x1, y0], [x1, y1]])
        eps = torch.finfo(xy.dtype).eps
        features = []
        for disk in self.source_disks:
            center = xy.new_tensor(disk.center)  # type: ignore[attr-defined]
            rm = float(disk.radius_m)  # type: ignore[attr-defined]
            rm2 = max(rm * rm, 1e-24)
            max_r2 = ((corners - center) ** 2).sum(dim=1).max()
            denom = (0.5 * torch.log1p(max_r2 / rm2)).clamp_min(eps)
            r2 = ((xy[:, :2] - center) ** 2).sum(dim=1, keepdim=True)
            log_d = 0.5 * torch.log1p(r2 / rm2)
            features.append(2.0 * log_d / denom - 1.0)
        return features

    def coordinates(self, region: RegionSpec, xy: Any, partition: str) -> Any:
        import torch

        local = self._local_tensor_coordinates(region, xy)
        if partition == "global":
            if local.shape[1] < 3:
                local = torch.cat([local, xy.new_zeros((len(xy), 3 - local.shape[1]))], dim=1)
            one_hot = xy.new_zeros((len(xy), len(self.regions)))
            one_hot[:, region.id] = 1.0
            return torch.cat([self.normalized_xy(xy), local, one_hot, *self._source_distance_features(xy, self.bounds)], dim=1)
        if region.shape.shape_type == "rectangle":
            return torch.cat([local, *self._source_distance_features(xy, region.shape.bounds)], dim=1)  # type: ignore[attr-defined]
        return local

    def unit_coordinates(self, region: RegionSpec, xy: Any, partition: str) -> Any:
        return (self.coordinates(region, xy, partition) + 1.0) / 2.0

    def planar_cuts(self) -> tuple[tuple[int, float], ...]:
        cuts: list[tuple[int, float]] = []
        x0, x1, y0, y1 = self.bounds
        for region in self.regions:
            s = region.shape
            if s.shape_type == "disk":
                cx, cy = s.center  # type: ignore[attr-defined]
                r = s.radius_m  # type: ignore[attr-defined]
                cuts.extend(((0, cx - r), (0, cx + r), (1, cy - r), (1, cy + r)))
            elif s.shape_type == "annulus":
                cx, cy = s.center  # type: ignore[attr-defined]
                for r in (s.inner_radius_m, s.outer_radius_m):  # type: ignore[attr-defined]
                    cuts.extend(((0, cx - r), (0, cx + r), (1, cy - r), (1, cy + r)))
            elif s.shape_type == "rectangle" and list(s.bounds) != [x0, x1, y0, y1]:  # type: ignore[attr-defined]
                a0, a1, b0, b1 = s.bounds  # type: ignore[attr-defined]
                cuts.extend(((0, a0), (0, a1), (1, b0), (1, b1)))
        for item in self.doc.get("interfaces", ()):
            if item["regions"][0] == item["regions"][1]:
                cuts.append((0 if item["axis"] == "x" else 1, float(item["position"])))
        return tuple(
            sorted(
                set(
                    (axis, float(pos))
                    for axis, pos in cuts
                    if self.bounds[axis * 2] < pos < self.bounds[axis * 2 + 1]
                )
            )
        )

    def _level_set_specs(self) -> tuple[tuple[Any, ...], ...]:
        specs: list[tuple[Any, ...]] = []
        x0, x1, y0, y1 = self.bounds
        for region in self.regions:
            s = region.shape
            if s.shape_type == "disk":
                specs.append(("circle", tuple(s.center), float(s.radius_m)))  # type: ignore[attr-defined]
            elif s.shape_type == "annulus":
                for r in (s.inner_radius_m, s.outer_radius_m):  # type: ignore[attr-defined]
                    specs.append(("circle", tuple(s.center), float(r)))  # type: ignore[attr-defined]
            elif s.shape_type == "rectangle" and list(s.bounds) != [x0, x1, y0, y1]:  # type: ignore[attr-defined]
                a0, a1, b0, b1 = s.bounds  # type: ignore[attr-defined]
                specs.extend((("x_left", float(a0)), ("x_right", float(a1)), ("y_bottom", float(b0)), ("y_top", float(b1))))
            elif s.shape_type == "polygon":
                a0, a1, b0, b1 = s._bbox()  # type: ignore[attr-defined]
                specs.extend((("x_left", float(a0)), ("x_right", float(a1)), ("y_bottom", float(b0)), ("y_top", float(b1))))
        for item in self.doc.get("interfaces", ()):
            if item["regions"][0] == item["regions"][1]:
                specs.append(("plane", 0 if item["axis"] == "x" else 1, float(item["position"])))
        return tuple(specs)

    def level_sets(self, xy: Any) -> Any:
        """Distancias con signo hacia todas las interfaces materiales del problema."""
        import torch

        columns = []
        for spec in self._level_set_specs():
            kind = spec[0]
            if kind == "circle":
                c = xy.new_tensor(spec[1])
                r_val = float(spec[2])
                eps_r = max(0.1 * r_val, 1e-6)
                r_reg = torch.sqrt(((xy[:, :2] - c) ** 2).sum(dim=1, keepdim=True) + eps_r * eps_r)
                columns.append(r_reg - math.hypot(r_val, eps_r))
            elif kind == "x_left":
                columns.append(xy[:, 0:1] - spec[1])
            elif kind == "x_right":
                columns.append(spec[1] - xy[:, 0:1])
            elif kind == "y_bottom":
                columns.append(xy[:, 1:2] - spec[1])
            elif kind == "plane":
                columns.append(xy[:, spec[1]:spec[1] + 1] - spec[2])
            else:
                columns.append(spec[1] - xy[:, 1:2])
        return torch.cat(columns, dim=1) if columns else xy.new_zeros((len(xy), 0))


class CaseCatalog:
    """Descubridor centralizado de casos JSON autocontenidos dentro de ``PINN_V4``."""

    def __init__(self, root: str | Path | None = None) -> None:
        if root is None:
            # Sube desde src/pinn_thermal/core/problem.py hasta la raíz PINN_V4 si existe
            pkg_root = Path(__file__).resolve().parents[3]
            self.root = pkg_root / "cases" if (pkg_root / "cases").is_dir() else Path.cwd() / "cases"
        else:
            self.root = Path(root)

    def list_entries(self) -> list[dict[str, Any]]:
        """Lista todos los casos organizados por categoría (``benchmarks``, ``thesis_curated``, etc.)."""
        entries: list[dict[str, Any]] = []
        if not self.root.is_dir():
            return entries
        for cat_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            for prob_file in sorted(cat_dir.glob("*/problem.json")):
                data = json.loads(prob_file.read_text(encoding="utf-8-sig"))
                entries.append(
                    {
                        "id": data["id"],
                        "category": cat_dir.name,
                        "title": data.get("title", data["id"]),
                        "family": data.get("family", "generic"),
                        "path": prob_file,
                    }
                )
        return entries

    def load(self, identifier: str | Path) -> ThermalProblem2D:
        """Carga un problema por ruta directa ``.json`` o por ``id`` dentro del catálogo."""
        path = Path(identifier)
        if path.is_file():
            return ThermalProblem2D.from_json(path)
        for entry in self.list_entries():
            if entry["id"] == str(identifier) or f"{entry['category']}/{entry['id']}" == str(identifier):
                return ThermalProblem2D.from_json(entry["path"])
        raise FileNotFoundError(f"No se encontró el caso {identifier!r} en el catálogo {self.root}.")


def load_problem(source: str | Path | dict[str, Any] | ThermalProblem2D) -> ThermalProblem2D:
    """Carga un ``ThermalProblem2D`` aceptando archivo, diccionario JSON o identificador de catálogo."""
    if isinstance(source, ThermalProblem2D):
        return source
    if isinstance(source, dict):
        return ThermalProblem2D.from_json(source)
    text = str(source)
    path = Path(source)
    if path.is_file() or text.lower().endswith(".json"):
        return ThermalProblem2D.from_json(path)
    return CaseCatalog().load(text)
