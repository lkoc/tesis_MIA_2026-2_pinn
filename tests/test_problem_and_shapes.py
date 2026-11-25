"""Pruebas de geometrías, esquema JSON, muestreo y aislamiento libre de PyTorch."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from pinn_thermal.core.problem import ThermalProblem2D
from pinn_thermal.core.schema import SCHEMA_V4, validate_problem
from pinn_thermal.core.shapes import AnnulusShape, DiskShape, PolygonShape, RectangleShape, Shape2D


def test_shapes_roundtrip_and_polygon() -> None:
    poly = PolygonShape(((0.0, 0.0), (2.0, 0.0), (1.0, 1.5)))
    assert poly.contains(np.array([1.0, 1.9]), np.array([0.5, 1.4])).tolist() == [True, False]
    assert Shape2D.from_dict(poly.to_dict()).to_dict() == poly.to_dict()

    for shp in (
        RectangleShape((-1.0, 1.0, -2.0, 0.0)),
        DiskShape((0.0, -1.0), 0.2),
        AnnulusShape((0.0, -1.0), 0.1, 0.3),
        poly,
    ):
        pts = shp.sample_local(32, np.random.default_rng(7), margin=1e-4)
        assert len(pts) == 32
        assert np.all(shp.contains(pts[:, 0], pts[:, 1], tolerance=1e-6))


def test_validator_rejects_negative_conductivity() -> None:
    bad = {
        "schema": SCHEMA_V4,
        "id": "bad_k",
        "domain": {"shape": "rectangle", "bounds": [0.0, 1.0, 0.0, 1.0]},
        "boundaries": {
            "bottom": {"type": "dirichlet", "value": 20.0},
            "top": {"type": "dirichlet", "value": 20.0},
            "left": {"type": "dirichlet", "value": 20.0},
            "right": {"type": "dirichlet", "value": 20.0},
        },
        "regions": [
            {
                "id": "soil",
                "priority": 0,
                "shape": {"type": "rectangle", "bounds": [0.0, 1.0, 0.0, 1.0]},
                "conductivity": -0.5,
                "source": 0.0,
            }
        ],
    }
    with pytest.raises(ValueError, match="positiva"):
        validate_problem(bad)


def test_core_geometry_and_fem_worker_are_torch_free(root_dir: Path) -> None:
    """Verifica en un subproceso limpio que geometría, campos y trabajador FEM no importen PyTorch."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root_dir / "src")
    code = (
        "import sys; "
        "import pinn_thermal.core.expressions, pinn_thermal.core.fields, pinn_thermal.core.shapes, "
        "pinn_thermal.core.schema, pinn_thermal.core.problem, pinn_thermal.core.manifest, "
        "pinn_thermal.fem.ufl_backend, pinn_thermal.fem.mesh_resolution, pinn_thermal.fem.worker; "
        "assert 'torch' not in sys.modules, f'PyTorch fue importado indebidamente: {list(sys.modules)}'; "
        "print('TORCH_FREE_OK')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    assert "TORCH_FREE_OK" in proc.stdout


def test_polygon_problem_construction() -> None:
    doc = {
        "schema": SCHEMA_V4,
        "id": "poly_demo",
        "domain": {"shape": "rectangle", "bounds": [-1.0, 1.0, -1.0, 1.0]},
        "boundaries": {
            "bottom": {"type": "dirichlet", "value": 20.0},
            "top": {"type": "dirichlet", "value": 20.0},
            "left": {"type": "neumann", "q_n_W_m2": 0.0},
            "right": {"type": "neumann", "q_n_W_m2": 0.0},
        },
        "regions": [
            {
                "id": "soil",
                "priority": 0,
                "shape": {"type": "rectangle", "bounds": [-1.0, 1.0, -1.0, 1.0]},
                "conductivity": 1.0,
                "source": 0.0,
            },
            {
                "id": "triangular_inclusion",
                "kind": "material",
                "priority": 1,
                "shape": {"type": "polygon", "vertices": [[-0.4, -0.4], [0.4, -0.4], [0.0, 0.5]]},
                "conductivity": 3.5,
                "source": 500.0,
            },
        ],
    }
    prob = ThermalProblem2D.from_json(doc)
    pts = prob.sample_region_interior(prob.regions[1], 40, 123)
    assert len(pts) == 40
    assert np.all(prob.owner(pts[:, 0], pts[:, 1]) == 1)


def test_shape_areas_thin_layer_scaling_and_smooth_distance_at_origin(catalog) -> None:
    """Verifica áreas exactas en regiones delgadas (sin 1e-12), escalas Fourier y suavidad C^inf en r=0."""
    import math
    import torch

    assert RectangleShape((-1.0, 2.0, 0.0, 4.0)).area() == pytest.approx(12.0)
    assert DiskShape((0.0, 0.0), 0.5).area() == pytest.approx(math.pi * 0.25)
    assert AnnulusShape((0.0, 0.0), 0.2, 0.5).area() == pytest.approx(math.pi * (0.25 - 0.04))
    assert PolygonShape(((0.0, 0.0), (2.0, 0.0), (0.0, 3.0))).area() == pytest.approx(3.0)

    prob = catalog.load("xlpe_discrete_layers")
    for reg in prob.regions:
        assert prob.areas[reg.id] > 1e-6, f"Área colapsada en {reg.name}: {prob.areas[reg.id]}"
        sc = prob.scales(reg)
        assert math.isfinite(sc["flux"]) and sc["flux"] < 1e5
        assert math.isfinite(sc["divergence"]) and sc["divergence"] < 1e9

    d0 = prob.source_disks[0]
    xy0 = torch.tensor([list(d0.center)], dtype=torch.float32, requires_grad=True)
    lvls = prob.level_sets(xy0)
    sfeats = torch.cat(prob._source_distance_features(xy0, prob.bounds), dim=1)
    for feat in (lvls, sfeats):
        assert torch.isfinite(feat).all()
        g = torch.autograd.grad(feat.sum(), xy0, create_graph=True)[0]
        h = torch.autograd.grad(g.sum(), xy0, retain_graph=True)[0]
        assert torch.isfinite(g).all()
        assert torch.isfinite(h).all()
