"""Validación de documentos ``pinn-v4-problem-1`` y conversión desde formatos heredados.

Cada error de validación nombra explícitamente el campo y la razón exacta del rechazo.
Este módulo es **100 % libre de PyTorch**.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

from pinn_thermal.core.fields import FieldSpec, evaluate_field_numpy, validate_field
from pinn_thermal.core.shapes import SHAPES, Shape2D

SCHEMA_V4: str = "pinn-v4-problem-1"
SCHEMA_V3: str = "pinn-v3-problem-1"
SUPPORTED_SCHEMAS: tuple[str, ...] = (SCHEMA_V4, SCHEMA_V3)

RECT_SIDES: tuple[str, ...] = ("bottom", "top", "left", "right")
ANNULUS_SIDES: tuple[str, ...] = ("inner", "outer")
BOUNDARY_TYPES: tuple[str, ...] = ("dirichlet", "neumann", "robin")
REGION_KEYS: frozenset[str] = frozenset(
    {"id", "kind", "priority", "shape", "conductivity", "source", "heat_capacity", "material", "metadata"}
)
TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        "schema",
        "id",
        "title",
        "description",
        "family",
        "units",
        "domain",
        "temperature_scale_K",
        "regions",
        "boundaries",
        "interfaces",
        "reference",
        "metadata",
        "annotations",
    }
)
COVERAGE_GRID: int = 201


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


class ProblemValidator:
    """Validador estricto para documentos físicos 2D en JSON."""

    @staticmethod
    def _validate_boundary(side: str, bc: Any) -> None:
        where = f"boundaries.{side}"
        _require(isinstance(bc, dict) and bc.get("type") in BOUNDARY_TYPES,
                 f"{where}.type debe ser uno de {BOUNDARY_TYPES}.")
        if bc["type"] == "dirichlet":
            _require(set(bc) == {"type", "value"}, f"{where}: dirichlet solo requiere 'type' y 'value'.")
            validate_field(bc["value"], f"{where}.value")
        elif bc["type"] == "neumann":
            _require(set(bc) == {"type", "q_n_W_m2"}, f"{where}: neumann solo requiere 'type' y 'q_n_W_m2'.")
            validate_field(bc["q_n_W_m2"], f"{where}.q_n_W_m2")
        else:
            _require(set(bc) == {"type", "h_W_m2K", "T_inf_C"},
                     f"{where}: robin requiere 'type', 'h_W_m2K' y 'T_inf_C'.")
            _require(_is_finite_number(bc["h_W_m2K"]) and float(bc["h_W_m2K"]) > 0,
                     f"{where}.h_W_m2K debe ser positivo y finito.")
            validate_field(bc["T_inf_C"], f"{where}.T_inf_C")

    @staticmethod
    def _validate_interfaces(doc: dict[str, Any], region_ids: set[str], bounds: list[float]) -> None:
        interfaces = doc.get("interfaces")
        if interfaces is None:
            return
        _require(isinstance(interfaces, list), "interfaces debe ser una lista JSON.")
        seen_ids: set[str] = set()
        x0, x1, y0, y1 = bounds
        for idx, item in enumerate(interfaces):
            where = f"interfaces[{idx}]"
            _require(isinstance(item, dict), f"{where} debe ser un objeto JSON.")
            _require({"id", "axis", "position", "regions"} <= set(item),
                     f"{where} requiere 'id', 'axis', 'position' y 'regions'.")
            iface_id = item["id"]
            _require(isinstance(iface_id, str) and iface_id.strip() != "", f"{where}.id no debe estar vacío.")
            _require(iface_id not in seen_ids, f"{where}.id duplicado: {iface_id!r}.")
            seen_ids.add(iface_id)
            _require(item["axis"] in ("x", "y"), f"{where}.axis debe ser 'x' o 'y'.")
            _require(_is_finite_number(item["position"]), f"{where}.position debe ser un número finito.")
            pos = float(item["position"])
            lo, hi = (x0, x1) if item["axis"] == "x" else (y0, y1)
            _require(lo < pos < hi, f"{where}.position ({pos}) debe caer estrictamente dentro de [{lo}, {hi}].")
            regs = item["regions"]
            _require(isinstance(regs, (list, tuple)) and len(regs) == 2,
                     f"{where}.regions debe tener exactamente 2 identificadores de región.")
            for rid in regs:
                _require(rid in region_ids, f"{where}.regions hace referencia a la región inexistente {rid!r}.")

    @staticmethod
    def _check_coverage_and_conductivity(doc: dict[str, Any]) -> None:
        """Verifica cobertura total del dominio y conductividad positiva en cada región propietaria."""
        domain = doc["domain"]
        if domain["shape"] == "annulus":
            angles = np.linspace(0.0, 2.0 * math.pi, COVERAGE_GRID, endpoint=False)
            radii = np.linspace(domain["inner_radius_m"], domain["outer_radius_m"], COVERAGE_GRID)
            radius, angle = np.meshgrid(radii, angles)
            x = (domain["center"][0] + radius * np.cos(angle)).ravel()
            y = (domain["center"][1] + radius * np.sin(angle)).ravel()
            span = max(float(domain["outer_radius_m"]) * 2.0, 1.0)
        else:
            x0, x1, y0, y1 = domain["bounds"]
            gx, gy = np.meshgrid(np.linspace(x0, x1, COVERAGE_GRID), np.linspace(y0, y1, COVERAGE_GRID))
            x, y = gx.ravel(), gy.ravel()
            span = max(float(x1 - x0), float(y1 - y0), 1.0)

        owner = np.full(x.shape, -1, dtype=int)
        tolerance = 1e-8 * span
        shapes = [Shape2D.from_dict(r["shape"]) for r in doc["regions"]]
        for index in sorted(range(len(doc["regions"])), key=lambda item: doc["regions"][item]["priority"]):
            owner[shapes[index].contains(x, y, tolerance=tolerance)] = index

        uncovered = int(np.count_nonzero(owner < 0))
        _require(uncovered == 0, f"{uncovered} puntos de la malla de verificación quedan sin región propietaria.")

        for index, region in enumerate(doc["regions"]):
            mine = owner == index
            if not np.any(mine):
                continue
            values = evaluate_field_numpy(region["conductivity"], x[mine], y[mine])
            _require(
                bool(np.all(np.isfinite(values)) and np.all(values > 0.0)),
                f"region {region['id']!r}: la conductividad térmica debe ser estrictamente positiva y finita.",
            )

    @classmethod
    def validate(cls, doc: dict[str, Any]) -> dict[str, Any]:
        """Valida un documento completo y lo devuelve normalizado; lanza ``ValueError`` si falla."""
        _require(isinstance(doc, dict), "El documento del problema debe ser un diccionario JSON.")
        schema = doc.get("schema")
        _require(schema in SUPPORTED_SCHEMAS, f"schema debe ser uno de {SUPPORTED_SCHEMAS}; recibido {schema!r}.")
        unknown_top = set(doc) - TOP_LEVEL_KEYS
        _require(not unknown_top, f"Claves de nivel superior no reconocidas: {sorted(unknown_top)}.")
        _require(isinstance(doc.get("id"), str) and doc["id"].strip() != "", "id debe ser una cadena no vacía.")

        domain = doc.get("domain", {})
        _require(isinstance(domain, dict) and domain.get("shape") in ("rectangle", "annulus"),
                 "domain.shape debe ser 'rectangle' o 'annulus'.")
        if domain["shape"] == "rectangle":
            bounds = domain.get("bounds", [])
            _require(isinstance(bounds, (list, tuple)) and len(bounds) == 4 and all(_is_finite_number(v) for v in bounds),
                     "El dominio rectangular necesita bounds=[x0, x1, y0, y1] finitos.")
        else:
            _require(set(domain) == {"shape", "center", "inner_radius_m", "outer_radius_m", "bounds"},
                     "El dominio anular requiere shape, center, inner_radius_m, outer_radius_m y bounds.")
            _require(len(domain["center"]) == 2 and all(_is_finite_number(v) for v in domain["center"]),
                     "domain.center debe ser [cx, cy].")
            _require(_is_finite_number(domain["inner_radius_m"]) and _is_finite_number(domain["outer_radius_m"]),
                     "Los radios del dominio anular deben ser finitos.")
            _require(0.0 < float(domain["inner_radius_m"]) < float(domain["outer_radius_m"]),
                     "El dominio anular requiere 0 < inner_radius_m < outer_radius_m.")
            bounds = domain["bounds"]
            _require(len(bounds) == 4 and all(_is_finite_number(v) for v in bounds),
                     "domain.bounds del dominio anular debe tener 4 valores finitos.")

        bounds_f = [float(v) for v in domain["bounds"]]
        _require(bounds_f[0] < bounds_f[1] and bounds_f[2] < bounds_f[3], "domain.bounds define un rectángulo vacío.")

        if "temperature_scale_K" in doc:
            scale = doc["temperature_scale_K"]
            _require(_is_finite_number(scale) and float(scale) > 0.0,
                     "temperature_scale_K debe ser un número finito positivo.")

        regions = doc.get("regions")
        _require(isinstance(regions, list) and len(regions) > 0, "regions debe ser una lista no vacía.")
        ids = [region.get("id") for region in regions if isinstance(region, dict)]
        _require(len(ids) == len(regions) and all(isinstance(r, str) and r.strip() != "" for r in ids),
                 "Cada región debe tener un 'id' único no vacío.")
        _require(len(set(ids)) == len(ids), "Los identificadores region.id deben ser únicos.")

        for region in regions:
            where = f"region {region['id']!r}"
            _require(set(region) <= REGION_KEYS and {"id", "priority", "shape", "conductivity"} <= set(region),
                     f"{where}: las claves obligatorias son id, priority, shape y conductivity (opcionales: kind, source, heat_capacity, metadata).")
            _require(isinstance(region["priority"], int) and not isinstance(region["priority"], bool),
                     f"{where}.priority debe ser un entero.")
            shape_obj = Shape2D.from_dict(region["shape"], f"{where}.shape")
            _require(shape_obj.inside_bounds(bounds_f), f"{where}.shape sale de los límites del dominio {bounds_f}.")
            if "kind" in region:
                _require(region["kind"] in ("soil", "conductor", "material"),
                         f"{where}.kind inválido; utilice 'soil', 'conductor' o 'material'.")
            validate_field(region["conductivity"], f"{where}.conductivity")
            validate_field(region.get("source", 0.0), f"{where}.source")
            if "heat_capacity" in region:
                validate_field(region["heat_capacity"], f"{where}.heat_capacity")

        _require(len({r["priority"] for r in regions}) == len(regions),
                 "Las prioridades region.priority deben ser únicas en todo el documento.")

        cls._validate_interfaces(doc, set(ids), bounds_f)

        boundaries = doc.get("boundaries")
        expected_sides = ANNULUS_SIDES if domain["shape"] == "annulus" else RECT_SIDES
        _require(isinstance(boundaries, dict) and set(boundaries) == set(expected_sides),
                 f"boundaries debe declarar exactamente los lados {expected_sides}.")
        for side in expected_sides:
            cls._validate_boundary(side, boundaries[side])

        reference = doc.get("reference", {})
        _require(isinstance(reference, dict) and set(reference) <= {"exact"},
                 "reference solo puede declarar la clave 'exact'.")
        if "exact" in reference:
            validate_field(reference["exact"], "reference.exact")

        cls._check_coverage_and_conductivity(doc)
        return doc


def validate_problem(doc: dict[str, Any]) -> dict[str, Any]:
    """Valida un documento de problema 2D según ``pinn-v4-problem-1`` / ``pinn-v3-problem-1``."""
    return ProblemValidator.validate(doc)


def convert_legacy_case(legacy: dict[str, Any]) -> dict[str, Any]:
    """Convierte un JSON de caso heredado (``pinn-v3-case-1`` o formato plano ``v2``) a ``pinn-v4-problem-1``.

    Transforma capas concéntricas de cables en regiones disjuntas ``disk + annulus`` con
    prioridades únicas, convierte estratos y parches en campos nativos de v4 y normaliza las
    condiciones de frontera a los cuatro lados del dominio rectangular.
    """
    if legacy.get("schema") in SUPPORTED_SCHEMAS:
        out = copy.deepcopy(legacy)
        out["schema"] = SCHEMA_V4
        return validate_problem(out)

    case_id = str(legacy.get("id", "caso_convertido"))
    if legacy.get("schema") == "pinn-v3-case-1":
        bounds = list(legacy["domain"]["bounds"])
        t_ref = float(legacy.get("temperature", {}).get("reference_C", 20.0))
        t_scale = float(legacy.get("temperature", {}).get("scale_K", 50.0))
        centers = list(legacy.get("conductors", {}).get("centers", []))
        layers = [
            (float(item["inner_radius_m"]), float(item["outer_radius_m"]), float(item["k_W_mK"]))
            for item in legacy.get("conductors", {}).get("layers", [])
        ]
        elec = legacy.get("electrical", {})
        power_w_m = elec.get("power_W_m")
        if power_w_m is None and "current_A" in elec and "R20_ohm_m" in elec:
            power_w_m = float(elec["current_A"]) ** 2 * float(elec["R20_ohm_m"])
        mat = legacy.get("material", {})
        k_base = float(mat.get("k_base_W_mK", 1.0))
        cond_spec = mat.get("conductivity")
        patch_spec = mat.get("patch")
        bands = mat.get("bands", [])
    else:
        # Formato v2 plano
        bounds = list(legacy["bounds"])
        t_ref = float(legacy.get("T0", 20.0))
        t_scale = float(legacy.get("scale", 50.0))
        centers = [list(c) for c in legacy.get("cables", [])]
        layers = [(float(r[0]), float(r[1]), float(r[2])) for r in legacy.get("layers", [])]
        power_w_m = legacy.get("power")
        if power_w_m is None and "current" in legacy and "R20" in legacy:
            power_w_m = float(legacy["current"]) ** 2 * float(legacy["R20"])
        k_base = float(legacy.get("k", 1.0))
        cond_spec = legacy.get("conductivity")
        patch_spec = legacy.get("patch")
        bands = legacy.get("bands", [])

    # Construcción del fondo del suelo
    regions: list[dict[str, Any]] = []
    priority = 0

    if isinstance(cond_spec, dict) and cond_spec.get("type") == "layers":
        smoothing = float(cond_spec.get("smoothing", 0.0))
        if smoothing == 0.0 and cond_spec.get("axis") in ("x", "y"):
            axis = cond_spec["axis"]
            ifaces = [float(v) for v in cond_spec["interfaces"]]
            vals = [float(v) for v in cond_spec["values"]]
            if axis == "y":
                edges = [bounds[2], *ifaces, bounds[3]]
                for idx, (y_lo, y_hi, kv) in enumerate(zip(edges, edges[1:], vals)):
                    regions.append(
                        {
                            "id": f"soil_layer_{idx}",
                            "kind": "soil",
                            "priority": priority,
                            "shape": {"type": "rectangle", "bounds": [bounds[0], bounds[1], y_lo, y_hi]},
                            "conductivity": kv,
                            "source": 0.0,
                        }
                    )
                    priority += 1
            else:
                edges = [bounds[0], *ifaces, bounds[1]]
                for idx, (x_lo, x_hi, kv) in enumerate(zip(edges, edges[1:], vals)):
                    regions.append(
                        {
                            "id": f"soil_layer_{idx}",
                            "kind": "soil",
                            "priority": priority,
                            "shape": {"type": "rectangle", "bounds": [x_lo, x_hi, bounds[2], bounds[3]]},
                            "conductivity": kv,
                            "source": 0.0,
                        }
                    )
                    priority += 1
        else:
            soil_k: Any = FieldSpec.from_dict(
                {
                    "layers": {
                        "axis": cond_spec["axis"],
                        "interfaces": cond_spec["interfaces"],
                        "values": cond_spec["values"],
                        "smoothing": smoothing,
                    }
                }
            ).to_dict()
            regions.append(
                {
                    "id": "background",
                    "kind": "soil",
                    "priority": priority,
                    "shape": {"type": "rectangle", "bounds": bounds},
                    "conductivity": soil_k,
                    "source": 0.0,
                }
            )
            priority += 1
    else:
        soil_k = k_base
        if isinstance(patch_spec, (list, tuple)) and len(patch_spec) >= 5:
            cx, cy, w, h, kp = [float(v) for v in patch_spec[:5]]
            eps = float(patch_spec[5]) if len(patch_spec) > 5 else 0.08
            soil_k = FieldSpec.from_dict(
                {
                    "patch": {
                        "base": k_base,
                        "center": [cx, cy],
                        "width": w,
                        "height": h,
                        "value": kp,
                        "smoothing": eps,
                    }
                }
            ).to_dict()
        regions.append(
            {
                "id": "background",
                "kind": "soil",
                "priority": priority,
                "shape": {"type": "rectangle", "bounds": bounds},
                "conductivity": soil_k,
                "source": 0.0,
            }
        )
        priority += 1

    # Bandas rectangulares opcionales
    for b_idx, band in enumerate(bands or ()):
        if isinstance(band, dict) and "bounds" in band and "k" in band:
            regions.append(
                {
                    "id": f"band_{b_idx}",
                    "kind": "soil",
                    "priority": priority,
                    "shape": {"type": "rectangle", "bounds": [float(v) for v in band["bounds"]]},
                    "conductivity": float(band["k"]),
                    "source": 0.0,
                }
            )
            priority += 1

    # Capas concéntricas de cables (disk + annulus)
    for c_idx, center in enumerate(centers):
        cx, cy = float(center[0]), float(center[1])
        for l_idx, (r_in, r_out, k_layer) in enumerate(layers):
            is_core = r_in <= 0.0
            area_core = math.pi * (r_out ** 2 - r_in ** 2)
            q_vol = float(power_w_m) / area_core if (is_core and power_w_m is not None) else 0.0
            shape: dict[str, Any] = (
                {"type": "disk", "center": [cx, cy], "radius_m": r_out}
                if is_core
                else {"type": "annulus", "center": [cx, cy], "inner_radius_m": r_in, "outer_radius_m": r_out}
            )
            regions.append(
                {
                    "id": f"cable_{c_idx}_layer_{l_idx}",
                    "kind": "conductor" if is_core else "material",
                    "priority": priority,
                    "shape": shape,
                    "conductivity": float(k_layer),
                    "source": q_vol,
                }
            )
            priority += 1

    boundaries = {side: {"type": "dirichlet", "value": t_ref} for side in RECT_SIDES}
    grad_x = legacy.get("outer_gradient_K_m")
    if _is_finite_number(grad_x) and float(grad_x) != 0.0:
        boundaries = {
            side: {"type": "dirichlet", "value": {"expression": f"{t_ref} + {float(grad_x)} * x"}}
            for side in RECT_SIDES
        }

    doc: dict[str, Any] = {
        "schema": SCHEMA_V4,
        "id": case_id,
        "title": str(legacy.get("title", case_id)),
        "family": "cable" if centers else "generic",
        "domain": {"shape": "rectangle", "bounds": bounds},
        "temperature_scale_K": t_scale,
        "regions": regions,
        "boundaries": boundaries,
        "reference": {},
        "metadata": {
            "converted_from": legacy.get("schema", "legacy-v2"),
            "provenance": legacy.get("provenance", {}),
            "electrical": legacy.get("electrical", {}),
        },
    }
    return validate_problem(doc)
