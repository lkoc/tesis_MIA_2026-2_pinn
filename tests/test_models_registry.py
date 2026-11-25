"""Pruebas de registro extensible, construcción y exactitud analítica de arquitecturas PINN."""

from __future__ import annotations

from pathlib import Path

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


def test_fbpinn_c2_partition_of_unity_and_bounded_initial_loss(
    catalog,
    mms_problem: ThermalProblem2D,
    cable_problem: ThermalProblem2D,
) -> None:
    """Verifica partición de la unidad C2 en todo el dominio (incluidos bordes) y pérdida inicial acotada."""
    from pinn_thermal.core.objective import PhysicsObjective

    for prob in (mms_problem, cable_problem):
        torch.manual_seed(0)
        model = ModelRegistry.build(prob, "FBPINN", width=32, depth=3)
        soil = prob.regions[0]
        assert soil.shape.shape_type == "rectangle"
        x0, x1, y0, y1 = soil.shape.bounds  # type: ignore[attr-defined]
        windows = model._region_windows[soil.id]

        # Grilla densa cubriendo interior y los 4 bordes externos + esquinas
        gx = torch.linspace(x0, x1, 33, dtype=torch.float32)
        gy = torch.linspace(y0, y1, 33, dtype=torch.float32)
        xx, yy = torch.meshgrid(gx, gy, indexing="ij")
        xy_grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1).requires_grad_(True)

        ws = [model._window_weight(xy_grid, win) for win in windows]
        wsum = sum(ws)
        assert float(wsum.min().item()) >= 0.999
        assert float(wsum.max().item()) <= 1.001

        # Verificar soporte compacto exacto en las esquinas globales
        corners = torch.tensor([[x0, y0], [x1, y1]], dtype=torch.float32)
        w_tile_00 = model._window_weight(corners, windows[0])  # ix=0, iy=0
        w_tile_11 = model._window_weight(corners, windows[-1])  # ix=1, iy=1
        assert float(w_tile_00[0].item()) == pytest.approx(1.0, abs=1e-6)
        assert float(w_tile_11[0].item()) == pytest.approx(0.0, abs=1e-6)
        assert float(w_tile_00[1].item()) == pytest.approx(0.0, abs=1e-6)
        assert float(w_tile_11[1].item()) == pytest.approx(1.0, abs=1e-6)

        # Verificar que el Laplaciano de las ventanas normalizadas esté acotado
        w0_norm = ws[0] / wsum
        grad0 = torch.autograd.grad(w0_norm.sum(), xy_grid, create_graph=True)[0]
        lap0 = (
            torch.autograd.grad(grad0[:, 0].sum(), xy_grid, retain_graph=True)[0][:, 0]
            + torch.autograd.grad(grad0[:, 1].sum(), xy_grid)[0][:, 1]
        )
        assert torch.isfinite(lap0).all()
        assert float(lap0.abs().max().item()) < 500.0

        # Verificar que la pérdida física inicial no explote a ~1e7
        obj = PhysicsObjective(model, seed=11)
        total, _, reg = obj()
        assert torch.isfinite(total)
        assert float(reg[soil.name].item()) < 1500.0
        assert float(total.item()) < 25000.0

    # Caso cable_disk (del smoke test): pérdida total inicial < 5000 (antes 5.71e7)
    cable_disk = catalog.load("cable_disk")
    torch.manual_seed(0)
    m_cd = ModelRegistry.build(cable_disk, "FBPINN", width=32, depth=3)
    tot_cd, parts_cd, _ = PhysicsObjective(m_cd, seed=11)()
    assert torch.isfinite(tot_cd)
    assert float(tot_cd.item()) < 5000.0
    assert float(parts_cd["pde"].item()) < 5000.0


def test_exact_boundary_enforces_dirichlet_neumann_and_robin(catalog) -> None:
    """Verifica imposición analítica exacta de bordes Dirichlet, Neumann y Robin e interfaces de cable."""
    from pinn_thermal.core.physics import derived_flux

    for case_id in ("mms_robin", "annulus", "coaxial_robin", "two_layer_slab"):
        prob = catalog.load(case_id)
        for arch in ("M1_EXACT", "CUSP_EXACT"):
            torch.manual_seed(0)
            model = ModelRegistry.build(prob, arch, width=16, depth=2).double()
            assert model.exact_boundary is True
            for seg in prob.sample_boundary_regions(16, rows=1):
                reg = prob.regions[seg["region"]]
                bc = prob.boundary_condition(seg["condition"])
                xy_b = torch.tensor(seg["xy"], dtype=torch.float64, requires_grad=True)
                n_b = torch.tensor(seg["normal"], dtype=torch.float64)
                t_b, _ = model.field(reg, xy_b)
                if bc["type"] == "dirichlet":
                    err = float((t_b - bc["value"](xy_b)).abs().max().item())
                else:
                    q_b = derived_flux(t_b, xy_b, prob.conductivity(reg, xy_b))
                    qn = (q_b * n_b).sum(1, keepdim=True)
                    if bc["type"] == "neumann":
                        err = float((qn - bc["value"](xy_b)).abs().max().item())
                    else:
                        err = float((qn - bc["h"] * (t_b - bc["T_inf"](xy_b))).abs().max().item())
                assert err < 1e-9, f"{arch} en {case_id} borde {seg['side']} ({bc['type']}): err={err}"

            if case_id == "coaxial_robin" and arch == "M1_EXACT":
                angles = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
                for ci in prob.circular_interfaces:
                    xy_c = torch.tensor(
                        np.c_[ci.center[0] + ci.radius * np.cos(angles), ci.center[1] + ci.radius * np.sin(angles)],
                        dtype=torch.float64,
                    )
                    tl, _ = model.field(prob.regions[ci.left], xy_c)
                    tr, _ = model.field(prob.regions[ci.right], xy_c)
                    assert float((tl - tr).abs().max().item()) < 1e-10


def test_all_models_smooth_at_conductor_center_and_healthy_init(catalog) -> None:
    """Verifica suavidad C^inf en r=0 y pérdidas iniciales sin explosiones en casos representativos."""
    from pinn_thermal.core.objective import PhysicsObjective

    archs = ("M0", "M0R", "M1", "M2", "CUSP", "CUSPR", "DCSNN", "FBPINN", "M1_EXACT", "CUSP_EXACT")
    for case_id in ("cable_disk", "xlpe_single", "xlpe_discrete_layers", "aras_flat"):
        prob = catalog.load(case_id)
        d0 = prob.source_disks[0]
        r_disk = next(r for r in prob.regions if r.shape is d0)
        for arch in archs:
            torch.manual_seed(0)
            model = ModelRegistry.build(prob, arch, width=16, depth=2)
            xy_c = torch.tensor([list(d0.center)], dtype=torch.float32, requires_grad=True)
            t_c, _ = model.field(r_disk, xy_c)
            g_c = torch.autograd.grad(t_c.sum(), xy_c, create_graph=True)[0]
            h_c = torch.autograd.grad(g_c[:, 0].sum(), xy_c, retain_graph=True)[0]
            assert torch.isfinite(t_c).all() and torch.isfinite(g_c).all() and torch.isfinite(h_c).all(), (
                f"Singularidad en r=0 para {arch} sobre {case_id}"
            )
            obj = PhysicsObjective(model, n_interior=64, n_layer=32, n_face=16, n_boundary=16, seed=11)
            tot, _, _ = obj()
            assert torch.isfinite(tot) and float(tot.item()) < 5e5, f"Pérdida inicial excesiva en {arch}/{case_id}: {tot}"
