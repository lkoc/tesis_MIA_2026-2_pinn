"""Configuración y fixtures compartidas de ``pytest`` para ``PINN_V4``."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

PINN_V4_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PINN_V4_ROOT / "src"
for candidate in (SRC_ROOT, PINN_V4_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from pinn_thermal.core.problem import CaseCatalog, ThermalProblem2D
from pinn_thermal.visualization.geometry_editor import CaseEditorHelper


@pytest.fixture(scope="session")
def root_dir() -> Path:
    return PINN_V4_ROOT


@pytest.fixture(scope="session")
def cases_root() -> Path:
    return PINN_V4_ROOT / "cases"


@pytest.fixture(scope="session")
def catalog(cases_root: Path) -> CaseCatalog:
    return CaseCatalog(cases_root)


@pytest.fixture(scope="session")
def mms_problem(catalog: CaseCatalog) -> ThermalProblem2D:
    return catalog.load("mms_constant")


@pytest.fixture(scope="session")
def cable_problem(catalog: CaseCatalog) -> ThermalProblem2D:
    return catalog.load("xlpe_backfill")


@pytest.fixture(scope="session")
def heterogeneous_problem() -> ThermalProblem2D:
    return CaseEditorHelper(CaseEditorHelper.template_two_layer_soil()).build_problem()
