"""Pruebas de registro extensible, construcción y exactitud analítica de arquitecturas PINN."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import pinn_thermal.models  # noqa: F401
from pinn_thermal.core.base_model import ModelRegistry
from pinn_thermal.core.problem import ThermalProblem2D

NATIVE_ARCHITECTURES = [
    name for name in ModelRegistry.list_models() if not name.startswith("DeepXDE_")
]


def test_registry_has_all_documented_models() -> None:
    expected = {
        "M0", "M0R", "M1", "M2",
        "CUSP", "CUSPR", "DCSNN", "FBPINN",
        "M1_EXACT", "M0_EXACT_BC", "M1_EXACT_BC", "DCSNN_EXACT_BC",
        "M0_EXACT", "CUSP_EXACT", "DCSNN_EXACT", "FBPINN_EXACT",
        "DeepXDE_PFNN", "DeepXDE_FNN",
    }
    assert expected <= set(ModelRegistry.list_models())
    for meta in ModelRegistry.describe_all():
        assert meta["authors"]
        assert meta["summary"]


@pytest.mark.parametrize("arch", NATIVE_ARCHITECTURES)
def test_all_native_models_forward_and_predict_on_cable_and_mms(
    arch: str,
    mms_problem: ThermalProblem2D,
    cable_problem: ThermalProblem2D,
) -> None:
    for prob in (mms_problem, cable_problem):
        model = ModelRegistry.build(prob, arch, width=16, depth=2, activation="tanh")
        pts = prob.sample_region_interior(prob.regions[0], 12, 7)
        res = model.predict(pts, with_flux=True)
        assert res["T"].shape == (12,)
        assert res["q"].shape == (12, 2)
        assert np.all(np.isfinite(res["T"]))
        assert np.all(np.isfinite(res["q"]))


def test_m1_exact_enforces_exact_cable_interface_continuity_and_dirichlet(catalog) -> None:
    prob = catalog.load("xlpe_single")
    model = ModelRegistry.build(prob, "M1_EXACT", width=16, depth=2).double()
    assert model.exact_constraints is True

    # 1. Verificar [T] = 0 exacto sobre el radio exacto de todas las interfaces circulares (precisión float64)
    angles = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    for ci in prob.circular_interfaces:
        xy_circle = np.c_[
            ci.center[0] + ci.radius * np.cos(angles),
            ci.center[1] + ci.radius * np.sin(angles),
        ]
        xy_t = torch.tensor(xy_circle, dtype=torch.float64)
        left_reg = prob.regions[ci.left]
        right_reg = prob.regions[ci.right]
        t_l, _ = model.field(left_reg, xy_t)
        t_r, _ = model.field(right_reg, xy_t)
        jump = float((t_l - t_r).abs().max().item())
        assert jump < 1e-10, f"Salto de temperatura inesperado en {ci.name}: {jump}"

    # 2. Verificar condición Dirichlet exacta sobre la frontera externa (rows=1)
    for seg in prob.sample_boundary_regions(16, rows=1):
        bc = prob.boundary_condition(seg["condition"])
        if bc["type"] == "dirichlet":
            xy_b = torch.tensor(seg["xy"], dtype=torch.float64)
            t_b, _ = model.field(prob.regions[seg["region"]], xy_b)
            t_target = bc["value"](xy_b)
            err = float((t_b - t_target).abs().max().item())
            assert err < 1e-10


def test_cusp_exact_enforces_dirichlet_on_mms(mms_problem: ThermalProblem2D) -> None:
    model = ModelRegistry.build(mms_problem, "CUSP_EXACT", width=16, depth=2)
    assert model.exact_constraints is True
    for seg in mms_problem.sample_boundary_regions(20, rows=1):
        bc = mms_problem.boundary_condition(seg["condition"])
        if bc["type"] == "dirichlet":
            xy_b = torch.tensor(seg["xy"], dtype=torch.float32)
            t_b, _ = model.field(mms_problem.regions[seg["region"]], xy_b)
            t_target = bc["value"](xy_b)
            assert float((t_b - t_target).abs().max().item()) < 1e-4
