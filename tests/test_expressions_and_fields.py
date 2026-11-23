"""Pruebas unitarias para el motor AST de expresiones y todas las modalidades de FieldSpec."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pinn_thermal.core.expressions import ExpressionEngine
from pinn_thermal.core.fields import (
    ConstantField,
    ExpressionField,
    FieldSpec,
    LayersField,
    PatchField,
    PiecewiseField,
    Table1DField,
    Table2DField,
)


def test_expression_engine_math_and_logic() -> None:
    x = np.array([-0.5, 0.25, 0.8])
    y = np.array([0.0, 0.5, 1.0])
    out = ExpressionEngine.evaluate(
        "where((0 <= x < 0.5) and (y >= 0.2), sin(pi*x) + 2.0, abs(x) + 1.0)",
        x,
        y,
        np,
    )
    expected = np.where((x >= 0) & (x < 0.5) & (y >= 0.2), np.sin(np.pi * x) + 2.0, np.abs(x) + 1.0)
    assert np.allclose(out, expected)

    xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    yt = torch.tensor(y, dtype=torch.float64, requires_grad=True)
    out_t = ExpressionEngine.evaluate("exp(-x**2) + cos(pi*y)", xt, yt, torch)
    out_t.sum().backward()
    assert torch.all(torch.isfinite(xt.grad)) and torch.all(torch.isfinite(yt.grad))


def test_expression_engine_rejects_unsafe_syntax() -> None:
    with pytest.raises(ValueError):
        ExpressionEngine.parse("__import__('os').system('echo nope')")
    with pytest.raises(ValueError):
        ExpressionEngine.parse("unknown_var + 1")


def test_all_field_modalities_numpy_and_torch() -> None:
    specs = [
        3.5,
        {"expression": "1.0 + 0.5*x + 0.25*y**2"},
        {"piecewise": [{"when": "x <= 0.0 and y < 0.5", "value": 0.6}, {"when": "x > 0.5", "value": {"expression": "1.2 + x"}}, {"default": 1.0}]},
        {"table": {"axis": "y", "coordinates": [-1.0, 0.0, 1.0], "values": [0.5, 1.0, 2.0], "interpolation": "linear"}},
        {"table": {"axis": "x", "coordinates": [-1.0, 0.0, 1.0], "values": [0.5, 1.0, 2.0], "interpolation": "step"}},
        {"table": {"axes": ["x", "y"], "x": [-1.0, 1.0], "y": [-1.0, 1.0], "values": [[1.0, 2.0], [3.0, 4.0]]}},
        {"layers": {"axis": "y", "interfaces": [0.0], "values": [0.5, 1.5], "smoothing": 0.05}},
        {"patch": {"base": 1.0, "center": [0.0, 0.0], "width": 0.8, "height": 0.8, "value": 2.5, "smoothing": 0.04}},
    ]
    x_np = np.array([-0.5, 0.0, 0.7])
    y_np = np.array([-0.4, 0.2, 0.8])
    x_t = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    y_t = torch.tensor(y_np, dtype=torch.float64, requires_grad=True)

    for raw in specs:
        field = FieldSpec.from_dict(raw)
        val_np = field.evaluate_numpy(x_np, y_np)
        val_t = field.evaluate_torch(x_t, y_t)
        assert val_np.shape == (3,)
        assert np.allclose(val_np, val_t.detach().numpy(), atol=1e-10)
        assert FieldSpec.from_dict(field.to_dict()).to_dict() == field.to_dict()

    # Verificar diferenciabilidad autograd de tabla 2D bilineal dentro de celda interior
    tbl2d = Table2DField.from_spec({"axes": ["x", "y"], "x": [0.0, 1.0], "y": [0.0, 1.0], "values": [[1.0, 3.0], [2.0, 6.0]]})
    xp = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
    yp = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
    val = tbl2d.evaluate_torch(xp, yp)
    assert abs(float(val.item()) - 3.0) < 1e-12
    val.sum().backward()
    assert abs(float(xp.grad.item()) - 3.0) < 1e-10  # d/dx en y=0.5: ((3-1)+(6-2))/2 = 3.0
