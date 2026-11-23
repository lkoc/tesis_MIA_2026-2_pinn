"""Asistente de edición y creación interactiva de casos 2D en formato JSON ``pinn-v4-problem-1``.

Diseñado para ser utilizado tanto desde la interfaz gráfica Streamlit como desde
cuadernos Jupyter o scripts de Python:
- Cargar catálogos de materiales térmicos autocontenidos (``data/materials.json``).
- Crear plantillas listas para usar (cable único, terna trifásica, suelo estratificado,
  placa heterogénea o corona coaxial).
- Construir y modificar campos heterogéneos en cualquiera de los 6 modos admitidos:
  constante, expresión ``(x, y)``, condiciones lógicas ``piecewise``, ``table`` 1D/2D,
  ``layers`` o ``patch``.
- Añadir/modificar anotaciones gráficas y validar todo antes de escribir a disco.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from pinn_thermal.core.manifest import write_json_atomic
from pinn_thermal.core.problem import ThermalProblem2D
from pinn_thermal.core.schema import SCHEMA_V4, validate_problem


def load_material_presets() -> dict[str, Any]:
    """Carga la biblioteca autocontenida de materiales térmicos en ``data/materials.json``."""
    data_path = Path(__file__).resolve().parents[1] / "data" / "materials.json"
    return json.loads(data_path.read_text(encoding="utf-8-sig"))


class CaseEditorHelper:
    """Constructor y modificador fluido de archivos JSON térmicos 2D."""

    def __init__(self, doc: dict[str, Any] | None = None) -> None:
        if doc is None:
            self.doc: dict[str, Any] = self.template_buried_single_cable()
        else:
            self.doc = copy.deepcopy(doc)

    @classmethod
    def from_problem(cls, problem_or_path: str | Path | ThermalProblem2D) -> CaseEditorHelper:
        if isinstance(problem_or_path, ThermalProblem2D):
            return cls(problem_or_path.doc)
        prob = ThermalProblem2D.from_json(problem_or_path)
        return cls(prob.doc)

    @classmethod
    def from_json_text(cls, text: str) -> CaseEditorHelper:
        data = json.loads(text)
        validate_problem(data)
        return cls(data)

    def to_json_text(self, indent: int = 2) -> str:
        return json.dumps(self.doc, indent=indent, ensure_ascii=False)

    def validate(self) -> list[str]:
        """Devuelve lista vacía si es válido o un mensaje descriptivo de error."""
        try:
            validate_problem(self.doc)
            return []
        except Exception as exc:
            return [str(exc)]

    def build_problem(self) -> ThermalProblem2D:
        return ThermalProblem2D.from_json(self.doc)

    def save(self, path: str | Path) -> Path:
        validate_problem(self.doc)
        target = Path(path)
        write_json_atomic(target, self.doc)
        return target

    # ------------------------------------------------------------------
    # Operaciones de alto nivel sobre regiones, heterogeneidad y anotaciones
    # ------------------------------------------------------------------

    def set_metadata(self, case_id: str, title: str, family: str | None = None, description: str = "") -> CaseEditorHelper:
        self.doc["schema"] = SCHEMA_V4
        self.doc["id"] = str(case_id)
        self.doc["title"] = str(title)
        if family is not None:
            self.doc["family"] = str(family)
        if description:
            self.doc["description"] = str(description)
        return self

    def set_rectangle_domain(self, x0: float, x1: float, y0: float, y1: float) -> CaseEditorHelper:
        self.doc["domain"] = {"shape": "rectangle", "bounds": [float(x0), float(x1), float(y0), float(y1)]}
        return self

    def set_boundary(
        self,
        side: str,
        bc_type: str,
        value: Any = 20.0,
        h_W_m2K: float = 10.0,
    ) -> CaseEditorHelper:
        boundaries = self.doc.setdefault("boundaries", {})
        if bc_type == "dirichlet":
            boundaries[side] = {"type": "dirichlet", "value": value}
        elif bc_type == "neumann":
            boundaries[side] = {"type": "neumann", "q_n_W_m2": value}
        elif bc_type == "robin":
            boundaries[side] = {"type": "robin", "h_W_m2K": float(h_W_m2K), "T_inf_C": value}
        else:
            raise ValueError(f"Tipo de frontera inválido: {bc_type!r}.")
        return self

    def upsert_region(
        self,
        region_id: str,
        kind: str,
        priority: int,
        shape: dict[str, Any],
        conductivity: Any,
        source: Any = 0.0,
        material: str = "",
    ) -> CaseEditorHelper:
        regions = self.doc.setdefault("regions", [])
        entry: dict[str, Any] = {
            "id": str(region_id),
            "kind": str(kind),
            "priority": int(priority),
            "shape": copy.deepcopy(shape),
            "conductivity": copy.deepcopy(conductivity),
            "source": copy.deepcopy(source),
        }
        if material:
            entry["material"] = material
        for idx, reg in enumerate(regions):
            if reg["id"] == entry["id"]:
                regions[idx] = entry
                return self
        regions.append(entry)
        return self

    def add_concentric_cable(
        self,
        prefix: str,
        center: tuple[float, float],
        layers: list[dict[str, Any]],
        priority_start: int = 10,
    ) -> CaseEditorHelper:
        """Agrega un cable completo formado por núcleo circular + coronas aislantes concéntricas."""
        r_prev = 0.0
        cx, cy = float(center[0]), float(center[1])
        for offset, layer in enumerate(layers):
            r_out = float(layer["outer_radius_m"])
            shape = (
                {"type": "disk", "center": [cx, cy], "radius_m": r_out}
                if r_prev == 0.0
                else {"type": "annulus", "center": [cx, cy], "inner_radius_m": r_prev, "outer_radius_m": r_out}
            )
            self.upsert_region(
                region_id=f"{prefix}_{layer['name']}",
                kind=layer.get("kind", "conductor" if r_prev == 0.0 else "material"),
                priority=int(priority_start + offset),
                shape=shape,
                conductivity=layer["conductivity"],
                source=layer.get("source", 0.0),
                material=layer.get("material", layer["name"]),
            )
            r_prev = r_out
        return self

    def add_annotation(self, x: float, y: float, text: str, kind: str = "note") -> CaseEditorHelper:
        notes = self.doc.setdefault("annotations", [])
        notes.append({"x": float(x), "y": float(y), "text": str(text), "kind": str(kind)})
        return self

    def clear_annotations(self) -> CaseEditorHelper:
        self.doc["annotations"] = []
        return self

    # ------------------------------------------------------------------
    # Constructores auxiliares para heterogeneidad espacial
    # ------------------------------------------------------------------

    @staticmethod
    def field_constant(value: float) -> float:
        return float(value)

    @staticmethod
    def field_expression(expr: str) -> dict[str, str]:
        return {"expression": str(expr)}

    @staticmethod
    def field_piecewise(rules: list[tuple[str, Any]], default: Any) -> dict[str, Any]:
        return {"piecewise": [{"when": cond, "value": val} for cond, val in rules] + [{"default": default}]}

    @staticmethod
    def field_table_1d(axis: str, coords: list[float], values: list[float], interpolation: str = "linear") -> dict[str, Any]:
        return {"table": {"axis": axis, "coordinates": list(coords), "values": list(values), "interpolation": interpolation}}

    @staticmethod
    def field_table_2d(xs: list[float], ys: list[float], matrix: list[list[float]], interpolation: str = "linear") -> dict[str, Any]:
        return {"table": {"axes": ["x", "y"], "x": list(xs), "y": list(ys), "values": matrix, "interpolation": interpolation}}

    @staticmethod
    def field_layers(axis: str, interfaces: list[float], values: list[Any], smoothing: float = 0.0) -> dict[str, Any]:
        return {"layers": {"axis": axis, "interfaces": list(interfaces), "values": list(values), "smoothing": float(smoothing)}}

    @staticmethod
    def field_patch(base: Any, center: tuple[float, float], width: float, height: float, value: Any, smoothing: float = 0.05) -> dict[str, Any]:
        return {
            "patch": {
                "base": base,
                "center": [float(center[0]), float(center[1])],
                "width": float(width),
                "height": float(height),
                "value": value,
                "smoothing": float(smoothing),
            }
        }

    # ------------------------------------------------------------------
    # Plantillas prediseñadas
    # ------------------------------------------------------------------

    @classmethod
    def template_buried_single_cable(cls) -> dict[str, Any]:
        """Plantilla estándar de cable XLPE monocore enterrado con relleno térmico."""
        return {
            "schema": SCHEMA_V4,
            "id": "nuevo_cable_subterraneo",
            "title": "Cable XLPE subterráneo en suelo con relleno térmico",
            "family": "cable",
            "units": {"length": "m", "temperature": "C", "conductivity": "W/(m*K)", "source": "W/m^3"},
            "domain": {"shape": "rectangle", "bounds": [-1.5, 1.5, -2.0, 0.0]},
            "boundaries": {
                "bottom": {"type": "dirichlet", "value": 20.0},
                "top": {"type": "robin", "h_W_m2K": 10.0, "T_inf_C": 25.0},
                "left": {"type": "neumann", "q_n_W_m2": 0.0},
                "right": {"type": "neumann", "q_n_W_m2": 0.0},
            },
            "regions": [
                {
                    "id": "soil",
                    "kind": "soil",
                    "priority": 0,
                    "shape": {"type": "rectangle", "bounds": [-1.5, 1.5, -2.0, 0.0]},
                    "conductivity": 1.0,
                    "source": 0.0,
                },
                {
                    "id": "backfill",
                    "kind": "soil",
                    "priority": 1,
                    "shape": {"type": "rectangle", "bounds": [-0.35, 0.35, -1.25, -0.75]},
                    "conductivity": 2.2,
                    "source": 0.0,
                },
                {
                    "id": "conductor",
                    "kind": "conductor",
                    "priority": 2,
                    "shape": {"type": "disk", "center": [0.0, -1.0], "radius_m": 0.018},
                    "conductivity": 380.0,
                    "source": 45000.0,
                },
                {
                    "id": "xlpe",
                    "kind": "material",
                    "priority": 3,
                    "shape": {"type": "annulus", "center": [0.0, -1.0], "inner_radius_m": 0.018, "outer_radius_m": 0.036},
                    "conductivity": 0.2857,
                    "source": 0.0,
                },
            ],
            "annotations": [
                {"x": 0.0, "y": -1.0, "text": "Cable XLPE (profundidad 1.0 m)", "kind": "note"}
            ],
        }

    @classmethod
    def template_two_layer_soil(cls) -> dict[str, Any]:
        """Plantilla con estratificación en suelo mediante tabla 1D y condiciones lógicas."""
        doc = cls.template_buried_single_cable()
        doc["id"] = "suelo_estratificado_tabla"
        doc["title"] = "Cable en suelo estratificado mediante tabla 1D"
        doc["regions"] = [
            doc["regions"][0] | {
                "conductivity": {
                    "table": {
                        "axis": "y",
                        "coordinates": [-2.0, -1.2, -0.8, 0.0],
                        "values": [0.7, 0.9, 1.5, 1.8],
                        "interpolation": "linear",
                    }
                }
            },
            doc["regions"][2] | {"priority": 1},
            doc["regions"][3] | {"priority": 2},
        ]
        return doc
