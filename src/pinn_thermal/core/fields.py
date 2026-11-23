"""Definición orientada a objetos de campos espaciales sobre ``(x, y)``.

Este módulo interpreta todas las modalidades declarables en los archivos JSON para
propiedades físicas (``conductivity``, ``source``, ``heat_capacity``), condiciones de
contorno y soluciones analíticas de referencia:

1. **Constante numérica**: ``1.5``.
2. **Expresión algebraica/lógica**: ``{"expression": "1 + 0.4*sin(2*pi*x)*cos(2*pi*y)"}``.
3. **Condiciones lógicas por tramos (``piecewise``)**:
   ``{"piecewise": [{"when": "x < 0.5 and y < 0", "value": 0.4}, {"default": 1.0}]}``.
4. **Tabla 1D** (lineal, vecino más cercano o escalonada):
   ``{"table": {"axis": "y", "coordinates": [-4, -1, 0], "values": [0.5, 1.0, 1.2], "interpolation": "linear"}}``.
5. **Tabla 2D** sobre rejilla cartesiana regular (interpolación bilineal o vecino más cercano):
   ``{"table": {"axes": ["x", "y"], "x": [...], "y": [...], "values": [[...], ...]}}``.
6. **Estratos paralelos (``layers``)** exactos o suavizados:
   ``{"layers": {"axis": "y", "interfaces": [-1.0], "values": [0.5, 1.0], "smoothing": 0.0}}``.
7. **Parche rectangular de heterogeneidad (``patch``)** exacto o suavizado:
   ``{"patch": {"base": 1.0, "center": [0.0, -0.7], "width": 0.5, "height": 0.5, "value": 2.0, "smoothing": 0.08}}``.

Este módulo es **libre de PyTorch en tiempo de importación**; los métodos tensores
importan ``torch`` solo cuando se invocan explícitamente o reciben ``torch`` como backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from pinn_thermal.core.expressions import ExpressionEngine


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _ensure_torch(torch_module: Any = None) -> Any:
    if torch_module is not None:
        return torch_module
    import torch

    return torch


def _searchsorted_1d_torch(knots: Any, coord: Any, torch_mod: Any, right: bool = True) -> Any:
    return torch_mod.clamp(
        torch_mod.searchsorted(knots, coord.contiguous(), right=right) - 1,
        0,
        len(knots) - 2,
    )


class FieldSpec(ABC):
    """Clase base abstracta para cualquier campo espacial escalar 2D."""

    @abstractmethod
    def evaluate_numpy(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Evalúa el campo en arrays de NumPy con la forma común de ``(x, y)``."""

    @abstractmethod
    def evaluate_torch(self, x: Any, y: Any, torch_mod: Any = None) -> Any:
        """Evalúa el campo como tensor de PyTorch preservando el grafo autograd."""

    def evaluate_ufl(self, x_expr: Any, y_expr: Any, ufl_backend: Any, fem_mod: Any, func_space: Any) -> Any:
        """Evalúa el campo dentro de una forma débil UFL de FEniCSx.

        Por defecto interpola numéricamente el campo sobre el espacio de elementos finitos
        ``func_space``, garantizando soporte uniforme incluso para tablas 2D o funciones
        arbitrarias. Las subclases algebraicas sobrescriben este método para evaluación
        simbólica exacta en puntos de cuadratura.
        """
        fn = fem_mod.Function(func_space)
        fn.interpolate(lambda z: self.evaluate_numpy(z[0], z[1]).ravel())
        return fn

    @abstractmethod
    def to_dict(self) -> Any:
        """Serializa el campo al formato JSON canónico."""

    @staticmethod
    def from_dict(spec: Any, where: str = "field") -> FieldSpec:
        """Fábrica que construye la instancia ``FieldSpec`` adecuada desde un valor o dict JSON."""
        if isinstance(spec, FieldSpec):
            return spec
        if isinstance(spec, bool):
            raise ValueError(f"{where}: los valores booleanos no son campos físicos válidos.")
        if isinstance(spec, (int, float)):
            _require(math.isfinite(float(spec)), f"{where}: el valor numérico debe ser finito.")
            return ConstantField(float(spec))
        _require(isinstance(spec, dict) and len(spec) == 1, (
            f"{where}: utilice un número o un objeto con una única clave entre "
            "{'expression', 'piecewise', 'table', 'layers', 'patch'}."
        ))
        key = next(iter(spec))
        if key == "expression":
            return ExpressionField.from_spec(spec["expression"], where)
        if key == "piecewise":
            return PiecewiseField.from_spec(spec["piecewise"], where)
        if key == "table":
            table = spec["table"]
            _require(isinstance(table, dict), f"{where}.table debe ser un objeto JSON.")
            if "axes" in table:
                return Table2DField.from_spec(table, where)
            return Table1DField.from_spec(table, where)
        if key == "layers":
            return LayersField.from_spec(spec["layers"], where)
        if key == "patch":
            return PatchField.from_spec(spec["patch"], where)
        raise ValueError(f"{where}: modalidad de campo desconocida {key!r}.")


@dataclass(frozen=True)
class ConstantField(FieldSpec):
    """Campo espacial uniforme constante."""

    value: float

    def __post_init__(self) -> None:
        _require(_is_finite_number(self.value), "ConstantField requiere un número finito.")

    def evaluate_numpy(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.full(np.shape(x), float(self.value), dtype=float)

    def evaluate_torch(self, x: Any, y: Any, torch_mod: Any = None) -> Any:
        return x * 0.0 + float(self.value)

    def evaluate_ufl(self, x_expr: Any, y_expr: Any, ufl_backend: Any, fem_mod: Any, func_space: Any) -> float:
        return float(self.value)

    def to_dict(self) -> float:
        return float(self.value)


@dataclass(frozen=True)
class ExpressionField(FieldSpec):
    """Campo definido mediante una expresión simbólica segura sobre ``(x, y)``."""

    expression: str

    @classmethod
    def from_spec(cls, text: Any, where: str = "field") -> ExpressionField:
        _require(isinstance(text, str) and text.strip() != "", f"{where}.expression debe ser una cadena no vacía.")
        try:
            ExpressionEngine.evaluate(text, np.array([0.5]), np.array([0.5]), np)
        except Exception as exc:
            raise ValueError(f"{where}: expresión inválida {text!r}: {exc}") from exc
        return cls(expression=text)

    def evaluate_numpy(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        shape = np.shape(x)
        evaluated = ExpressionEngine.evaluate(self.expression, x, y, np)
        return np.broadcast_to(np.asarray(evaluated, dtype=float), shape).copy()

    def evaluate_torch(self, x: Any, y: Any, torch_mod: Any = None) -> Any:
        tm = _ensure_torch(torch_mod)
        return x * 0.0 + ExpressionEngine.evaluate(self.expression, x, y, tm)

    def evaluate_ufl(self, x_expr: Any, y_expr: Any, ufl_backend: Any, fem_mod: Any, func_space: Any) -> Any:
        return ExpressionEngine.evaluate(self.expression, x_expr, y_expr, ufl_backend)

    def to_dict(self) -> dict[str, str]:
        return {"expression": self.expression}


@dataclass(frozen=True)
class PiecewiseField(FieldSpec):
    """Campo definido mediante reglas lógicas secuenciales ``if / elif / else``.

    Formato JSON::

        {
          "piecewise": [
            {"when": "x < 0.5 and y <= -1.0", "value": 0.45},
            {"when": "sqrt(x**2 + (y+0.7)**2) < 0.3", "value": {"expression": "1.2 + 0.1*x"}},
            {"default": 1.0}
          ]
        }

    La última regla debe declarar ``"default"`` (o una condición ``"when"`` con un
    valor de respaldo implícito ``0.0`` si no se incluye, aunque se exige ``default``
    explícito para mayor claridad).
    """

    rules: tuple[tuple[str, FieldSpec], ...]
    default: FieldSpec

    @classmethod
    def from_spec(cls, spec: Any, where: str = "field") -> PiecewiseField:
        _require(isinstance(spec, list) and len(spec) >= 1, f"{where}.piecewise debe ser una lista no vacía.")
        rules: list[tuple[str, FieldSpec]] = []
        default_field: FieldSpec | None = None
        for idx, item in enumerate(spec):
            _require(isinstance(item, dict), f"{where}.piecewise[{idx}] debe ser un objeto JSON.")
            if "default" in item and set(item) == {"default"}:
                _require(default_field is None, f"{where}.piecewise solo admite un bloque 'default'.")
                default_field = FieldSpec.from_dict(item["default"], f"{where}.piecewise[{idx}].default")
            elif {"when", "value"} <= set(item) and set(item) <= {"when", "value"}:
                cond = item["when"]
                _require(isinstance(cond, str) and cond.strip() != "", f"{where}.piecewise[{idx}].when vacío.")
                try:
                    ExpressionEngine.evaluate(cond, np.array([0.5]), np.array([0.5]), np)
                except Exception as exc:
                    raise ValueError(f"{where}.piecewise[{idx}].when inválida {cond!r}: {exc}") from exc
                val = FieldSpec.from_dict(item["value"], f"{where}.piecewise[{idx}].value")
                rules.append((cond, val))
            else:
                raise ValueError(
                    f"{where}.piecewise[{idx}] debe contener {{'when': ..., 'value': ...}} o {{'default': ...}}."
                )
        _require(default_field is not None, f"{where}.piecewise debe finalizar con una entrada {{'default': ...}}.")
        return cls(rules=tuple(rules), default=default_field)

    def evaluate_numpy(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        acc = self.default.evaluate_numpy(x, y)
        for cond, branch in reversed(self.rules):
            mask = np.asarray(ExpressionEngine.evaluate(cond, x, y, np), dtype=bool)
            acc = np.where(mask, branch.evaluate_numpy(x, y), acc)
        return np.asarray(acc, dtype=float)

    def evaluate_torch(self, x: Any, y: Any, torch_mod: Any = None) -> Any:
        tm = _ensure_torch(torch_mod)
        acc = self.default.evaluate_torch(x, y, tm)
        for cond, branch in reversed(self.rules):
            mask = ExpressionEngine.evaluate(cond, x, y, tm)
            if mask.dtype != tm.bool:
                mask = mask > 0.5
            acc = tm.where(mask, branch.evaluate_torch(x, y, tm), acc)
        return acc

    def evaluate_ufl(self, x_expr: Any, y_expr: Any, ufl_backend: Any, fem_mod: Any, func_space: Any) -> Any:
        acc = self.default.evaluate_ufl(x_expr, y_expr, ufl_backend, fem_mod, func_space)
        for cond, branch in reversed(self.rules):
            mask = ExpressionEngine.evaluate(cond, x_expr, y_expr, ufl_backend)
            acc = ufl_backend.where(mask, branch.evaluate_ufl(x_expr, y_expr, ufl_backend, fem_mod, func_space), acc)
        return acc

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        items: list[dict[str, Any]] = [
            {"when": cond, "value": branch.to_dict()} for cond, branch in self.rules
        ]
        items.append({"default": self.default.to_dict()})
        return {"piecewise": items}


@dataclass(frozen=True)
class Table1DField(FieldSpec):
    """Campo interpolado desde una tabla 1D sobre el eje ``x`` o el eje ``y``."""

    axis: str
    coordinates: tuple[float, ...]
    values: tuple[float, ...]
    interpolation: str = "linear"

    _INTERPOLATIONS = ("linear", "nearest", "step")

    @classmethod
    def from_spec(cls, table: dict[str, Any], where: str = "field") -> Table1DField:
        allowed_keys = {"axis", "coordinates", "values", "interpolation"}
        _require({"axis", "coordinates", "values"} <= set(table) and set(table) <= allowed_keys, (
            f"{where}.table debe definir axis ('x' o 'y'), coordinates, values y opcionalmente interpolation."
        ))
        axis = table["axis"]
        _require(axis in ("x", "y"), f"{where}.table.axis debe ser 'x' o 'y'.")
        interp = table.get("interpolation", "linear")
        _require(interp in cls._INTERPOLATIONS, f"{where}.table.interpolation debe ser una de {cls._INTERPOLATIONS}.")
        coords = table["coordinates"]
        vals = table["values"]
        _require(
            isinstance(coords, (list, tuple)) and isinstance(vals, (list, tuple)) and len(coords) >= 2 and len(coords) == len(vals),
            f"{where}.table requiere al menos dos pares coordenada-valor de igual longitud.",
        )
        _require(all(_is_finite_number(c) for c in coords) and all(_is_finite_number(v) for v in vals),
                 f"{where}.table solo admite valores numéricos finitos.")
        coords_f = tuple(float(c) for c in coords)
        vals_f = tuple(float(v) for v in vals)
        _require(all(a < b for a, b in zip(coords_f, coords_f[1:])),
                 f"{where}.table.coordinates debe ser estrictamente creciente.")
        return cls(axis=axis, coordinates=coords_f, values=vals_f, interpolation=interp)

    def evaluate_numpy(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        coord = np.asarray(x if self.axis == "x" else y, dtype=float)
        knots = np.asarray(self.coordinates, dtype=float)
        vals = np.asarray(self.values, dtype=float)
        if self.interpolation == "linear":
            return np.interp(coord, knots, vals)
        clamped = np.clip(coord, knots[0], knots[-1])
        idx = np.clip(np.searchsorted(knots, clamped, side="right") - 1, 0, len(knots) - 2)
        if self.interpolation == "step":
            # En el extremo superior conserva el último valor de tabla
            use_last = coord >= knots[-1]
            out = vals[idx]
            return np.where(use_last, vals[-1], out)
        # Vecino más cercano
        d_left = np.abs(clamped - knots[idx])
        d_right = np.abs(knots[idx + 1] - clamped)
        return np.where(d_left <= d_right, vals[idx], vals[idx + 1])

    def evaluate_torch(self, x: Any, y: Any, torch_mod: Any = None) -> Any:
        tm = _ensure_torch(torch_mod)
        original_shape = x.shape
        coord = (x if self.axis == "x" else y).reshape(-1)
        knots = tm.as_tensor(self.coordinates, dtype=x.dtype, device=x.device)
        vals = tm.as_tensor(self.values, dtype=x.dtype, device=x.device)
        clamped = tm.minimum(tm.maximum(coord, knots[0]), knots[-1])
        idx = _searchsorted_1d_torch(knots, clamped, tm, right=True)
        if self.interpolation == "linear":
            t = (clamped - knots[idx]) / (knots[idx + 1] - knots[idx])
            out = vals[idx] + t * (vals[idx + 1] - vals[idx])
        elif self.interpolation == "step":
            out = tm.where(coord >= knots[-1], vals[-1], vals[idx])
        else:
            d_left = (clamped - knots[idx]).abs()
            d_right = (knots[idx + 1] - clamped).abs()
            out = tm.where(d_left <= d_right, vals[idx], vals[idx + 1])
        return out.reshape(original_shape)

    def to_dict(self) -> dict[str, Any]:
        spec: dict[str, Any] = {
            "axis": self.axis,
            "coordinates": list(self.coordinates),
            "values": list(self.values),
        }
        if self.interpolation != "linear":
            spec["interpolation"] = self.interpolation
        return {"table": spec}


@dataclass(frozen=True)
class Table2DField(FieldSpec):
    """Campo interpolado desde una tabla cartesiana 2D ``values[iy][ix]``."""

    x_coords: tuple[float, ...]
    y_coords: tuple[float, ...]
    values: tuple[tuple[float, ...], ...]
    interpolation: str = "linear"

    @classmethod
    def from_spec(cls, table: dict[str, Any], where: str = "field") -> Table2DField:
        allowed_keys = {"axes", "x", "y", "values", "interpolation"}
        _require({"axes", "x", "y", "values"} <= set(table) and set(table) <= allowed_keys,
                 f"{where}.table 2D requiere axes=['x','y'], x, y y matriz values.")
        _require(list(table["axes"]) == ["x", "y"], f"{where}.table.axes debe ser ['x', 'y'].")
        interp = table.get("interpolation", "linear")
        _require(interp in ("linear", "nearest"), f"{where}.table.interpolation 2D debe ser 'linear' o 'nearest'.")
        xs, ys, matrix = table["x"], table["y"], table["values"]
        _require(len(xs) >= 2 and len(ys) >= 2, f"{where}.table 2D requiere al menos 2 puntos por eje.")
        _require(all(_is_finite_number(v) for v in xs) and all(_is_finite_number(v) for v in ys),
                 f"{where}.table 2D solo admite coordenadas finitas.")
        xs_f = tuple(float(v) for v in xs)
        ys_f = tuple(float(v) for v in ys)
        _require(all(a < b for a, b in zip(xs_f, xs_f[1:])) and all(a < b for a, b in zip(ys_f, ys_f[1:])),
                 f"{where}.table 2D requiere coordenadas x e y estrictamente crecientes.")
        _require(isinstance(matrix, (list, tuple)) and len(matrix) == len(ys_f),
                 f"{where}.table.values debe tener {len(ys_f)} filas (eje y).")
        rows: list[tuple[float, ...]] = []
        for r_idx, row in enumerate(matrix):
            _require(isinstance(row, (list, tuple)) and len(row) == len(xs_f),
                     f"{where}.table.values[{r_idx}] debe tener {len(xs_f)} columnas (eje x).")
            _require(all(_is_finite_number(v) for v in row), f"{where}.table 2D valores deben ser finitos.")
            rows.append(tuple(float(v) for v in row))
        return cls(x_coords=xs_f, y_coords=ys_f, values=tuple(rows), interpolation=interp)

    def evaluate_numpy(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        shape = xa.shape
        x_flat, y_flat = xa.ravel(), ya.ravel()
        kx = np.asarray(self.x_coords, dtype=float)
        ky = np.asarray(self.y_coords, dtype=float)
        grid = np.asarray(self.values, dtype=float)

        xc = np.clip(x_flat, kx[0], kx[-1])
        yc = np.clip(y_flat, ky[0], ky[-1])
        ix = np.clip(np.searchsorted(kx, xc, side="right") - 1, 0, len(kx) - 2)
        iy = np.clip(np.searchsorted(ky, yc, side="right") - 1, 0, len(ky) - 2)

        tx = (xc - kx[ix]) / (kx[ix + 1] - kx[ix])
        ty = (yc - ky[iy]) / (ky[iy + 1] - ky[iy])

        if self.interpolation == "nearest":
            ixn = np.where(tx <= 0.5, ix, ix + 1)
            iyn = np.where(ty <= 0.5, iy, iy + 1)
            return grid[iyn, ixn].reshape(shape)

        v00 = grid[iy, ix]
        v10 = grid[iy, ix + 1]
        v01 = grid[iy + 1, ix]
        v11 = grid[iy + 1, ix + 1]
        out = (1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v10 + (1 - tx) * ty * v01 + tx * ty * v11
        return out.reshape(shape)

    def evaluate_torch(self, x: Any, y: Any, torch_mod: Any = None) -> Any:
        tm = _ensure_torch(torch_mod)
        shape = x.shape
        xf, yf = x.reshape(-1), y.reshape(-1)
        kx = tm.as_tensor(self.x_coords, dtype=x.dtype, device=x.device)
        ky = tm.as_tensor(self.y_coords, dtype=x.dtype, device=x.device)
        grid = tm.as_tensor(self.values, dtype=x.dtype, device=x.device)

        xc = tm.minimum(tm.maximum(xf, kx[0]), kx[-1])
        yc = tm.minimum(tm.maximum(yf, ky[0]), ky[-1])
        ix = _searchsorted_1d_torch(kx, xc, tm, right=True)
        iy = _searchsorted_1d_torch(ky, yc, tm, right=True)

        tx = (xc - kx[ix]) / (kx[ix + 1] - kx[ix])
        ty = (yc - ky[iy]) / (ky[iy + 1] - ky[iy])

        if self.interpolation == "nearest":
            ixn = tm.where(tx <= 0.5, ix, ix + 1)
            iyn = tm.where(ty <= 0.5, iy, iy + 1)
            return grid[iyn, ixn].reshape(shape)

        v00 = grid[iy, ix]
        v10 = grid[iy, ix + 1]
        v01 = grid[iy + 1, ix]
        v11 = grid[iy + 1, ix + 1]
        out = (1.0 - tx) * (1.0 - ty) * v00 + tx * (1.0 - ty) * v10 + (1.0 - tx) * ty * v01 + tx * ty * v11
        return out.reshape(shape)

    def to_dict(self) -> dict[str, Any]:
        spec: dict[str, Any] = {
            "axes": ["x", "y"],
            "x": list(self.x_coords),
            "y": list(self.y_coords),
            "values": [list(r) for r in self.values],
        }
        if self.interpolation != "linear":
            spec["interpolation"] = self.interpolation
        return {"table": spec}


@dataclass(frozen=True)
class LayersField(FieldSpec):
    """Campo estratificado a lo largo de ``x`` o ``y`` (exacto o suavizado con tanh)."""

    axis: str
    interfaces: tuple[float, ...]
    values: tuple[float, ...]
    smoothing: float = 0.0

    @classmethod
    def from_spec(cls, spec: dict[str, Any], where: str = "field") -> LayersField:
        _require(isinstance(spec, dict) and {"axis", "interfaces", "values"} <= set(spec)
                 and set(spec) <= {"axis", "interfaces", "values", "smoothing"},
                 f"{where}.layers requiere axis, interfaces, values y smoothing opcional.")
        axis = spec["axis"]
        _require(axis in ("x", "y"), f"{where}.layers.axis debe ser 'x' o 'y'.")
        ifaces = spec["interfaces"]
        vals = spec["values"]
        _require(isinstance(ifaces, (list, tuple)) and isinstance(vals, (list, tuple))
                 and len(ifaces) >= 1 and len(vals) == len(ifaces) + 1,
                 f"{where}.layers requiere N interfaces y N+1 valores.")
        _require(all(_is_finite_number(z) for z in ifaces) and all(_is_finite_number(v) for v in vals),
                 f"{where}.layers requiere números finitos.")
        ifaces_f = tuple(float(z) for z in ifaces)
        vals_f = tuple(float(v) for v in vals)
        _require(all(a < b for a, b in zip(ifaces_f, ifaces_f[1:])),
                 f"{where}.layers.interfaces debe ser estrictamente creciente.")
        smooth = float(spec.get("smoothing", 0.0))
        _require(_is_finite_number(smooth) and smooth >= 0.0, f"{where}.layers.smoothing debe ser >= 0.")
        return cls(axis=axis, interfaces=ifaces_f, values=vals_f, smoothing=smooth)

    def _evaluate_backend(self, x: Any, y: Any, backend: Any) -> Any:
        coord = x if self.axis == "x" else y
        acc = 0.0 * coord + self.values[0]
        for z, prev_val, next_val in zip(self.interfaces, self.values[:-1], self.values[1:]):
            if self.smoothing > 0.0:
                acc = acc + (next_val - prev_val) * 0.5 * (1.0 + backend.tanh((coord - z) / self.smoothing))
            else:
                acc = backend.where(coord > z, 0.0 * coord + next_val, acc)
        return acc

    def evaluate_numpy(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.asarray(self._evaluate_backend(np.asarray(x, dtype=float), np.asarray(y, dtype=float), np), dtype=float)

    def evaluate_torch(self, x: Any, y: Any, torch_mod: Any = None) -> Any:
        tm = _ensure_torch(torch_mod)
        return self._evaluate_backend(x, y, tm)

    def evaluate_ufl(self, x_expr: Any, y_expr: Any, ufl_backend: Any, fem_mod: Any, func_space: Any) -> Any:
        return self._evaluate_backend(x_expr, y_expr, ufl_backend)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layers": {
                "axis": self.axis,
                "interfaces": list(self.interfaces),
                "values": list(self.values),
                "smoothing": float(self.smoothing),
            }
        }


@dataclass(frozen=True)
class PatchField(FieldSpec):
    """Campo de fondo con un parche rectangular de valor distinto (suavizado o exacto)."""

    base: float
    center: tuple[float, float]
    width: float
    height: float
    value: float
    smoothing: float = 0.08

    @classmethod
    def from_spec(cls, spec: dict[str, Any], where: str = "field") -> PatchField:
        req = {"base", "center", "width", "height", "value"}
        _require(isinstance(spec, dict) and req <= set(spec) and set(spec) <= req | {"smoothing"},
                 f"{where}.patch requiere base, center [cx, cy], width, height, value y smoothing opcional.")
        _require(_is_finite_number(spec["base"]) and _is_finite_number(spec["value"]),
                 f"{where}.patch base y value deben ser números finitos.")
        center = spec["center"]
        _require(isinstance(center, (list, tuple)) and len(center) == 2 and all(_is_finite_number(v) for v in center),
                 f"{where}.patch.center debe ser [cx, cy].")
        _require(_is_finite_number(spec["width"]) and float(spec["width"]) > 0
                 and _is_finite_number(spec["height"]) and float(spec["height"]) > 0,
                 f"{where}.patch width y height deben ser positivos.")
        smooth = float(spec.get("smoothing", 0.08))
        _require(_is_finite_number(smooth) and smooth >= 0.0, f"{where}.patch.smoothing debe ser >= 0.")
        return cls(
            base=float(spec["base"]),
            center=(float(center[0]), float(center[1])),
            width=float(spec["width"]),
            height=float(spec["height"]),
            value=float(spec["value"]),
            smoothing=smooth,
        )

    def _evaluate_backend(self, x: Any, y: Any, backend: Any) -> Any:
        cx, cy = self.center
        hw, hh = 0.5 * self.width, 0.5 * self.height
        if self.smoothing > 0.0:
            wx = 0.5 * (backend.tanh((x - (cx - hw)) / self.smoothing) - backend.tanh((x - (cx + hw)) / self.smoothing))
            wy = 0.5 * (backend.tanh((y - (cy - hh)) / self.smoothing) - backend.tanh((y - (cy + hh)) / self.smoothing))
            return self.base + (self.value - self.base) * wx * wy + 0.0 * x
        inside = backend.logical_and(
            backend.logical_and(x >= cx - hw, x <= cx + hw),
            backend.logical_and(y >= cy - hh, y <= cy + hh),
        ) if hasattr(backend, "logical_and") else ((x >= cx - hw) & (x <= cx + hw) & (y >= cy - hh) & (y <= cy + hh))
        return backend.where(inside, 0.0 * x + self.value, 0.0 * x + self.base)

    def evaluate_numpy(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.asarray(self._evaluate_backend(np.asarray(x, dtype=float), np.asarray(y, dtype=float), np), dtype=float)

    def evaluate_torch(self, x: Any, y: Any, torch_mod: Any = None) -> Any:
        tm = _ensure_torch(torch_mod)
        return self._evaluate_backend(x, y, tm)

    def evaluate_ufl(self, x_expr: Any, y_expr: Any, ufl_backend: Any, fem_mod: Any, func_space: Any) -> Any:
        return self._evaluate_backend(x_expr, y_expr, ufl_backend)

    def to_dict(self) -> dict[str, Any]:
        return {
            "patch": {
                "base": float(self.base),
                "center": [float(self.center[0]), float(self.center[1])],
                "width": float(self.width),
                "height": float(self.height),
                "value": float(self.value),
                "smoothing": float(self.smoothing),
            }
        }


# Funciones auxiliares retrocompatibles para validar y evaluar campos directamente desde dicts o FieldSpec
def validate_field(spec: Any, where: str = "field") -> FieldSpec:
    """Valida una especificación de campo y devuelve su instancia ``FieldSpec``."""
    return FieldSpec.from_dict(spec, where=where)


def evaluate_field_numpy(spec: Any, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Evalúa cualquier especificación de campo en coordenadas NumPy ``(x, y)``."""
    return FieldSpec.from_dict(spec).evaluate_numpy(x, y)


def evaluate_field_torch(spec: Any, x: Any, y: Any, torch_mod: Any = None) -> Any:
    """Evalúa cualquier especificación de campo como tensor PyTorch sobre ``(x, y)``."""
    return FieldSpec.from_dict(spec).evaluate_torch(x, y, torch_mod=torch_mod)


def evaluate_field(spec: Any, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Alias retrocompatible de ``evaluate_field_numpy``."""
    return evaluate_field_numpy(spec, x, y)
