"""Backend simbólico compatible con UFL de FEniCSx para evaluar expresiones ``(x, y)``.

Este módulo es estrictamente **libre de PyTorch** y traduce las funciones matemáticas
aceptadas por :class:`pinn_thermal.core.expressions.ExpressionEngine` al árbol de
expresiones simbólicas ``ufl`` dentro del proceso trabajador de elementos finitos.
"""

from __future__ import annotations

from typing import Any


class UFLMathBackend:
    """Adaptador con interfaz tipo NumPy/Torch que construye nodos UFL de FEniCSx."""

    def __init__(self, ufl_module: Any) -> None:
        self._ufl = ufl_module
        self.pi = float(3.14159265358979323846)
        self.e = float(2.71828182845904523536)

    def sin(self, v: Any) -> Any:
        return self._ufl.sin(v)

    def cos(self, v: Any) -> Any:
        return self._ufl.cos(v)

    def tan(self, v: Any) -> Any:
        return self._ufl.tan(v)

    def tanh(self, v: Any) -> Any:
        return self._ufl.tanh(v)

    def sinh(self, v: Any) -> Any:
        return 0.5 * (self._ufl.exp(v) - self._ufl.exp(-v))

    def cosh(self, v: Any) -> Any:
        return 0.5 * (self._ufl.exp(v) + self._ufl.exp(-v))

    def exp(self, v: Any) -> Any:
        return self._ufl.exp(v)

    def log(self, v: Any) -> Any:
        return self._ufl.ln(v)

    def sqrt(self, v: Any) -> Any:
        return self._ufl.sqrt(v)

    def abs(self, v: Any) -> Any:
        return abs(v)

    def where(self, cond: Any, a: Any, b: Any) -> Any:
        return self._ufl.conditional(cond, a, b)

    def maximum(self, a: Any, b: Any) -> Any:
        return self._ufl.conditional(self._ufl.ge(a, b), a, b)

    def minimum(self, a: Any, b: Any) -> Any:
        return self._ufl.conditional(self._ufl.le(a, b), a, b)

    def logical_and(self, a: Any, b: Any) -> Any:
        return self._ufl.And(a, b)

    def logical_or(self, a: Any, b: Any) -> Any:
        return self._ufl.Or(a, b)

    def logical_not(self, a: Any) -> Any:
        return self._ufl.Not(a)
