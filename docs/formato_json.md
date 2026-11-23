# Manual Completo del Formato JSON (`pinn-v4-problem-1`)

Todos los modelos PINN, el solucionador FEniCSx y las herramientas gráficas leen íntegramente el problema térmico 2D desde un único archivo `.json`. Este documento permite redactar o editar casos directamente en cualquier editor de texto plano o mediante `CaseEditorHelper` / Streamlit.

---

## 1. Estructura de Nivel Superior

```json
{
  "schema": "pinn-v4-problem-1",
  "id": "cable_suelo_heterogeneo",
  "title": "Cable XLPE en suelo estratificado con relleno térmico",
  "description": "Texto descriptivo opcional.",
  "family": "cable",
  "units": {"length": "m", "temperature": "C", "conductivity": "W/(m*K)", "source": "W/m^3"},
  "temperature_scale_K": 50.0,
  "domain": { "shape": "rectangle", "bounds": [-2.0, 2.0, -3.0, 0.0] },
  "boundaries": { ... },
  "regions": [ ... ],
  "interfaces": [ ... ],
  "reference": { "exact": ... },
  "annotations": [ {"x": 0.0, "y": -1.0, "text": "Cable principal", "kind": "note"} ],
  "metadata": {}
}
```

### Reglas de claves obligatorias y opcionales
- **`schema`** *(obligatoria)*: `"pinn-v4-problem-1"` (acepta también `"pinn-v3-problem-1"` por retrocompatibilidad).
- **`id`** *(obligatoria)*: identificador único legible sin espacios.
- **`title`**, **`description`**, **`units`**, **`temperature_scale_K`**, **`interfaces`**, **`reference`**, **`annotations`**, **`metadata`**: opcionales.
- **`family`** *(opcional)*: `"cable"`, `"mms"`, `"annulus"` o `"generic"`. Si se omite, el núcleo la deduce automáticamente inspeccionando fuentes y geometrías.

---

## 2. Definición del Dominio (`domain`) y Fronteras (`boundaries`)

### Dominio Rectangular
```json
"domain": { "shape": "rectangle", "bounds": [x0, x1, y0, y1] },
"boundaries": {
  "bottom": { "type": "dirichlet", "value": 20.0 },
  "top":    { "type": "robin", "h_W_m2K": 10.0, "T_inf_C": 25.0 },
  "left":   { "type": "neumann", "q_n_W_m2": 0.0 },
  "right":  { "type": "neumann", "q_n_W_m2": 0.0 }
}
```

### Dominio Anular
```json
"domain": {
  "shape": "annulus",
  "center": [0.0, 0.0],
  "inner_radius_m": 0.1,
  "outer_radius_m": 1.0,
  "bounds": [-1.0, 1.0, -1.0, 1.0]
},
"boundaries": {
  "inner": { "type": "dirichlet", "value": 70.0 },
  "outer": { "type": "robin", "h_W_m2K": 15.0, "T_inf_C": 20.0 }
}
```

> **Convención de signos para Neumann**: `q_n_W_m2` es el flujo saliente positivo $-k \nabla T \cdot \mathbf{n}$ respecto a la normal exterior $\mathbf{n}$. Para aislamiento adiabático perfecto use `0.0`. Cualquier campo de frontera (`value`, `q_n_W_m2`, `T_inf_C`) acepta **cualquiera** de los formatos espaciales descritos en la Sección 4.

---

## 3. Regiones, Geometrías y Regla de Prioridad (`regions`)

Cada objeto dentro de `regions` representa un material o zona:

```json
{
  "id": "aislamiento_xlpe",
  "kind": "material",
  "priority": 3,
  "material": "xlpe",
  "shape": { "type": "annulus", "center": [0.0, -1.0], "inner_radius_m": 0.018, "outer_radius_m": 0.036 },
  "conductivity": 0.2857,
  "source": 0.0
}
```

- **`kind`**: `"soil"`, `"conductor"` o `"material"`.
- **`priority`**: entero arbitrario; **donde dos figuras se superponen, la región con mayor `priority` es propietaria del punto**. Para cables multicapa use núcleo `disk` (`0 -> r_0`) seguido de coronas `annulus` (`r_0 -> r_1`, `r_1 -> r_2`) con prioridades crecientes sobre el suelo base (`priority: 0`).
- **Tipos de `shape` soportados**:
  1. `"rectangle"`: `{"type": "rectangle", "bounds": [x0, x1, y0, y1]}`
  2. `"disk"`: `{"type": "disk", "center": [cx, cy], "radius_m": r}`
  3. `"annulus"`: `{"type": "annulus", "center": [cx, cy], "inner_radius_m": ri, "outer_radius_m": ro}`
  4. `"polygon"`: `{"type": "polygon", "vertices": [[x1, y1], [x2, y2], [x3, y3], ...]}` (polígono cerrado $\ge 3$ vértices en sentido horario o antihorario).

---

## 4. Formatos de Heterogeneidad Espacial para `conductivity`, `source` y `boundaries`

Cualquier parámetro físico acepta **7 modalidades autocontenidas**:

### 4.1. Constante Numérica
```json
"conductivity": 1.25
```

### 4.2. Expresión Analítica en `(x, y)`
```json
"conductivity": { "expression": "1.0 + 0.4 * exp(-((x)**2 + (y + 1.0)**2) / 0.5) * (1 + 0.1*sin(pi*x))" }
```
Variables permitidas: `x`, `y`, `pi`, `e`.  
Funciones permitidas: `sin`, `cos`, `tan`, `tanh`, `sinh`, `cosh`, `exp`, `log`, `sqrt`, `abs`, `where(cond, a, b)`, `min`, `max`.  
Comparadores y operadores lógicos: `+`, `-`, `*`, `/`, `**`, `<`, `<=`, `>`, `>=`, `==`, `!=`, `and`, `or`, `not`, `&`, `|`, `~`.

### 4.3. Condiciones Lógicas Secuenciales (`piecewise`)
```json
"conductivity": {
  "piecewise": [
    { "when": "y <= -1.2 and abs(x) <= 0.5", "value": 2.4 },
    { "when": "sqrt(x**2 + (y + 1.0)**2) < 0.35", "value": { "expression": "1.8 + 0.2*x" } },
    { "default": 1.0 }
  ]
}
```
Las reglas se evalúan en orden estricto (primera coincidencia gana); los valores de cada rama pueden ser constantes o cualquier otro campo anidado válido.

### 4.4. Tabla 1D (`table` en `x` o `y`)
```json
"conductivity": {
  "table": {
    "axis": "y",
    "coordinates": [-3.0, -1.5, -0.5, 0.0],
    "values": [0.6, 0.9, 1.4, 1.8],
    "interpolation": "linear"
  }
}
```
Modos de `"interpolation"`: `"linear"` (por defecto), `"nearest"` o `"step"` (escalonada por tramos).

### 4.5. Tabla 2D Bilineal sobre Grilla Regular (`table` 2D)
```json
"conductivity": {
  "table": {
    "axes": ["x", "y"],
    "x": [-1.5, 0.0, 1.5],
    "y": [-2.0, -1.0, 0.0],
    "values": [
      [0.8, 0.9, 0.8],
      [1.1, 1.3, 1.1],
      [1.5, 1.7, 1.5]
    ],
    "interpolation": "linear"
  }
}
```
> Importante: `values` tiene tamaño `len(y)` filas por `len(x)` columnas (`values[iy][ix]`), con `x` e `y` estrictamente crecientes.

### 4.6. Estratos Paralelos (`layers`)
```json
"conductivity": {
  "layers": {
    "axis": "y",
    "interfaces": [-1.2, -0.4],
    "values": [0.6, 1.1, 1.8],
    "smoothing": 0.05
  }
}
```
`len(values)` debe ser igual a `len(interfaces) + 1`. Si `smoothing > 0`, usa transición $\tanh$ diferenciable de espesor `smoothing` (m).

### 4.7. Parche Localizado (`patch`)
```json
"conductivity": {
  "patch": {
    "base": 1.0,
    "center": [0.0, -0.7],
    "width": 0.6,
    "height": 0.4,
    "value": 2.3,
    "smoothing": 0.06
  }
}
```

---

## 5. Validación desde Terminal o Python

```bash
# Validar cualquier archivo JSON editado a mano antes de entrenar
pinn-thermal validate ruta/a/mi_problema.json
```

En Python:
```python
from pinn_thermal.core.schema import validate_problem
import json

validate_problem(json.loads(open("mi_problema.json", encoding="utf-8-sig").read()))
```
