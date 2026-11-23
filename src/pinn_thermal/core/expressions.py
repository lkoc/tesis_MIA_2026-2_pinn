"""Motor seguro de expresiones algebraicas y lógicas sobre coordenadas (x, y).

Este módulo es **100 % libre de PyTorch** a nivel de importación, de modo que puede
ejecutarse tanto con NumPy, como con PyTorch o con UFL (FEniCSx dentro de WSL2/Linux)
mediante inyección de backend.

Características principales:
- Análisis sintáctico mediante el AST de Python (sin uso de ``eval`` ni ``exec``).
- Operadores aritméticos estándar: ``+``, ``-``, ``*``, ``/``, ``**``, signos unarios.
- Comparaciones simples y encadenadas: ``<``, ``<=``, ``>``, ``>=``, ``==``, ``!=``.
- Lógica booleana: ``and``, ``or``, ``not``, ``&``, ``|``, ``~``.
- Funciones matemáticas permitidas: ``sin``, ``cos``, ``tan``, ``tanh``, ``sinh``,
  ``cosh``, ``exp``, ``log``, ``sqrt``, ``abs``, ``where``, ``min``, ``max``.
- Constantes permitidas: ``pi``, ``e``. Variables espaciales: ``x``, ``y``.
"""

from __future__ import annotations

import ast
import math
from functools import lru_cache
from typing import Any

# Conjunto cerrado de funciones matemáticas y condicionales admitidas en expresiones.
ALLOWED_FUNCTIONS: frozenset[str] = frozenset(
    {
        "sin",
        "cos",
        "tan",
        "tanh",
        "sinh",
        "cosh",
        "exp",
        "log",
        "sqrt",
        "abs",
        "where",
        "min",
        "max",
    }
)

# Constantes simbólicas válidas dentro de una expresión.
ALLOWED_CONSTANTS: frozenset[str] = frozenset({"pi", "e"})

# Variables espaciales bidimensionales soportadas.
ALLOWED_VARIABLES: frozenset[str] = frozenset({"x", "y"})

# Tipos de nodos AST permitidos para evitar ejecución arbitraria de código.
_ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Call,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Not,
    ast.Invert,
    ast.And,
    ast.Or,
    ast.BitAnd,
    ast.BitOr,
    ast.Compare,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Eq,
    ast.NotEq,
)


def _backend_logical_and(backend: Any, left: Any, right: Any) -> Any:
    """Combina dos máscaras booleanas con conjunción lógica según el backend."""
    if hasattr(backend, "logical_and"):
        return backend.logical_and(left, right)
    return left & right


def _backend_logical_or(backend: Any, left: Any, right: Any) -> Any:
    """Combina dos máscaras booleanas con disyunción lógica según el backend."""
    if hasattr(backend, "logical_or"):
        return backend.logical_or(left, right)
    return left | right


def _backend_logical_not(backend: Any, operand: Any) -> Any:
    """Niega una máscara booleana según el backend."""
    if hasattr(backend, "logical_not"):
        return backend.logical_not(operand)
    return ~operand


class ExpressionEngine:
    """Analizador y evaluador portable de expresiones simbólicas sobre ``(x, y)``.

    La misma cadena de texto puede evaluarse sobre arrays de NumPy, tensores de PyTorch
    con diferenciación automática o coordenadas espaciales de UFL en FEniCSx.
    """

    MAX_EXPRESSION_LENGTH: int = 4000

    @classmethod
    @lru_cache(maxsize=256)
    def parse(cls, expression: str) -> ast.AST:
        """Valida sintaxis y seguridad de una expresión y devuelve el nodo raíz AST.

        Args:
            expression: Cadena de texto con la expresión matemática o lógica.

        Returns:
            Nodo raíz del árbol AST listo para evaluarse repetidamente.

        Raises:
            ValueError: Si la expresión excede la longitud permitida, contiene nodos
                prohibidos, constantes no finitas o identificadores desconocidos.
        """
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError("La expresión debe ser una cadena de texto no vacía.")
        if len(expression) > cls.MAX_EXPRESSION_LENGTH:
            raise ValueError(
                f"La expresión supera la longitud máxima permitida ({cls.MAX_EXPRESSION_LENGTH} caracteres)."
            )

        try:
            tree = ast.parse(expression, mode="eval").body
        except SyntaxError as exc:
            raise ValueError(f"Error de sintaxis en expresión {expression!r}: {exc}") from exc

        allowed_names = ALLOWED_VARIABLES | ALLOWED_CONSTANTS | ALLOWED_FUNCTIONS
        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED_AST_NODES):
                raise ValueError(f"Sintaxis no permitida en expresión: {type(node).__name__}")
            if isinstance(node, ast.Constant):
                if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                    raise ValueError(f"Solo se admiten constantes numéricas finitas, no {node.value!r}")
                if not math.isfinite(node.value):
                    raise ValueError("Las constantes numéricas deben ser finitas.")
            elif isinstance(node, ast.Name):
                if node.id not in allowed_names:
                    raise ValueError(f"Identificador no reconocido en expresión: {node.id!r}")
            elif isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCTIONS:
                    raise ValueError("Solo se permiten llamadas directas a funciones matemáticas autorizadas.")
                if node.keywords:
                    raise ValueError("No se admiten argumentos con nombre en funciones de expresiones.")
                if node.func.id == "where" and len(node.args) != 3:
                    raise ValueError("La función where(...) requiere exactamente 3 argumentos.")
                if node.func.id in ("min", "max") and len(node.args) < 2:
                    raise ValueError(f"La función {node.func.id}(...) requiere al menos 2 argumentos.")
                if node.func.id not in ("where", "min", "max") and len(node.args) != 1:
                    raise ValueError(f"La función {node.func.id}(...) requiere exactamente 1 argumento.")
        return tree

    @classmethod
    def evaluate(cls, expression: str, x: Any, y: Any, backend: Any) -> Any:
        """Evalúa una expresión validada en ``(x, y)`` usando el backend provisto.

        Args:
            expression: Cadena válida según ``ExpressionEngine.parse``.
            x: Coordenada horizontal (escalar, ndarray, Tensor o expresión UFL).
            y: Coordenada vertical con la misma forma que ``x``.
            backend: Módulo o adaptador con operaciones vectorizadas (``np``, ``torch``
                o ``UFLBackend``).

        Returns:
            Resultado evaluado con soporte completo para broadcasting sobre ``(x, y)``.
        """
        tree = cls.parse(expression)

        def _eval(node: ast.AST) -> Any:
            if isinstance(node, ast.Constant):
                return float(node.value)
            if isinstance(node, ast.Name):
                if node.id == "x":
                    return x
                if node.id == "y":
                    return y
                if node.id == "pi":
                    return math.pi
                if node.id == "e":
                    return math.e
                raise ValueError(f"Identificador inesperado: {node.id}")
            if isinstance(node, ast.UnaryOp):
                operand = _eval(node.operand)
                if isinstance(node.op, ast.USub):
                    return -operand
                if isinstance(node.op, ast.UAdd):
                    return +operand
                if isinstance(node.op, (ast.Not, ast.Invert)):
                    return _backend_logical_not(backend, operand)
                raise ValueError(f"Operador unario no soportado: {type(node.op).__name__}")
            if isinstance(node, ast.BinOp):
                left, right = _eval(node.left), _eval(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div):
                    return left / right
                if isinstance(node.op, ast.Pow):
                    return left ** right
                if isinstance(node.op, ast.BitAnd):
                    return _backend_logical_and(backend, left, right)
                if isinstance(node.op, ast.BitOr):
                    return _backend_logical_or(backend, left, right)
                raise ValueError(f"Operador binario no soportado: {type(node.op).__name__}")
            if isinstance(node, ast.BoolOp):
                values = [_eval(item) for item in node.values]
                acc = values[0]
                combiner = _backend_logical_and if isinstance(node.op, ast.And) else _backend_logical_or
                for item in values[1:]:
                    acc = combiner(backend, acc, item)
                return acc
            if isinstance(node, ast.Compare):
                # Soporta tanto comparaciones simples como encadenadas (p. ej. 0 <= x <= 1)
                left_val = _eval(node.left)
                acc = None
                for op, comp in zip(node.ops, node.comparators):
                    right_val = _eval(comp)
                    if isinstance(op, ast.Lt):
                        term = left_val < right_val
                    elif isinstance(op, ast.LtE):
                        term = left_val <= right_val
                    elif isinstance(op, ast.Gt):
                        term = left_val > right_val
                    elif isinstance(op, ast.GtE):
                        term = left_val >= right_val
                    elif isinstance(op, ast.Eq):
                        term = left_val == right_val
                    elif isinstance(op, ast.NotEq):
                        term = left_val != right_val
                    else:
                        raise ValueError(f"Comparador no soportado: {type(op).__name__}")
                    acc = term if acc is None else _backend_logical_and(backend, acc, term)
                    left_val = right_val
                return acc
            if isinstance(node, ast.Call):
                fname = node.func.id
                args = [_eval(arg) for arg in node.args]
                if fname == "where":
                    return backend.where(args[0], args[1] + 0 * x, args[2] + 0 * x)
                if fname in ("min", "max"):
                    acc = args[0] + 0 * x
                    for nxt in args[1:]:
                        nxt_b = nxt + 0 * x
                        cond = (acc <= nxt_b) if fname == "min" else (acc >= nxt_b)
                        acc = backend.where(cond, acc, nxt_b)
                    return acc
                if fname == "abs" and not hasattr(backend, "abs"):
                    return abs(args[0] + 0 * x)
                return getattr(backend, fname)(args[0] + 0 * x)
            raise ValueError(f"Nodo AST no soportado: {type(node).__name__}")

        result = _eval(tree)
        return result + 0 * x


# Funciones de conveniencia compatibles con la interfaz histórica de v3
def parse(expression: str) -> ast.AST:
    """Analiza y almacena en caché una expresión válida sobre ``(x, y)``."""
    return ExpressionEngine.parse(expression)


def evaluate(expression: str, x: Any, y: Any, backend: Any) -> Any:
    """Evalúa ``expression`` sobre las coordenadas ``(x, y)`` con el ``backend`` indicado."""
    return ExpressionEngine.evaluate(expression, x, y, backend)
