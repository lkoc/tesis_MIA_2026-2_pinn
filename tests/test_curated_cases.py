"""Pruebas sobre los 29 casos curados del catálogo y autocontención total de PINN_V4."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pinn_thermal.core.problem import CaseCatalog, ThermalProblem2D
from pinn_thermal.core.schema import SCHEMA_V4, validate_problem


def test_all_29_curated_cases_valid_and_sampleable(catalog: CaseCatalog) -> None:
    entries = catalog.list_entries()
    assert len(entries) == 29
    categories = {e["category"] for e in entries}
    assert categories == {"benchmarks", "thesis_curated"}

    for entry in entries:
        doc = json.loads(entry["path"].read_text(encoding="utf-8-sig"))
        assert doc["schema"] == SCHEMA_V4
        validate_problem(doc)
        prob = ThermalProblem2D.from_json(entry["path"])
        for reg in prob.regions:
            pts = prob.sample_region_interior(reg, 8, rng=19)
            assert pts.shape == (8, 2)


def test_fixed_cases_mms_layered_y_and_xlpe_discrete_layers(catalog: CaseCatalog) -> None:
    # 1. Consistencia MMS exacta de mms_layered_y (-div(k grad T) == Q)
    mms = catalog.load("mms_layered_y")
    for px, py in [(0.3, 0.25), (0.7, 0.75)]:
        h = 1e-5
        pts = np.array([[px, py], [px + h, py], [px - h, py], [px, py + h], [px, py - h]])
        t = mms.reference(pts)
        k = mms.field_numpy("conductivity", np.array([px]), np.array([py]))[0]
        lap = (t[1] - 2 * t[0] + t[2]) / h ** 2 + (t[3] - 2 * t[0] + t[4]) / h ** 2
        q_doc = mms.field_numpy("source", np.array([px]), np.array([py]))[0]
        assert abs(-k * lap - q_doc) < 1e-3

    # 2. Estratificación restaurada en xlpe_discrete_layers
    xlpe = catalog.load("xlpe_discrete_layers")
    k_bot = float(xlpe.field_numpy("conductivity", np.array([-2.0]), np.array([-2.5]))[0])
    k_top = float(xlpe.field_numpy("conductivity", np.array([-2.0]), np.array([-0.5]))[0])
    assert abs(k_bot - 0.5) < 1e-12
    assert abs(k_top - 1.0) < 1e-12


def test_zero_external_legacy_imports_in_pinn_v4(root_dir: Path) -> None:
    """Garantiza que ningún archivo Python de PINN_V4 importe código externo de pinn_v3 o pinn_v2."""
    forbidden = ("from pinn_v3", "import pinn_v3", "from pinn_v2", "import pinn_v2")
    for folder in ("src", "frontend", "tests"):
        for py_file in (root_dir / folder).rglob("*.py"):
            if py_file.name == "test_curated_cases.py":
                continue
            text = py_file.read_text(encoding="utf-8-sig")
            for pattern in forbidden:
                assert pattern not in text, f"Referencia externa prohibida {pattern!r} en {py_file}"
