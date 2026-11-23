# Catálogo Curado de Casos — `PINN_V4`

Todos los problemas están definidos en el formato autocontenido **`pinn-v4-problem-1`** y validados automáticamente por `pinn_thermal.core.schema.ProblemValidator` sin dependencias externas.

## Organización de carpetas

- **`benchmarks/` (12 casos)**: problemas de verificación matemática (MMS con solución exacta analítica, dominios anulares, interfaces de alto contraste, placas de dos capas y demostración de tablas 1D/2D + lógica condicional `piecewise`).
- **`thesis_curated/` (17 casos)**: casos curados de cables subterráneos de media y alta tensión (configuraciones monocore XLPE, trifásicos planos y trébol simétricos ARAS, zanjas de relleno térmico, condiciones Robin/Dirichlet, y los escenarios Kim et al. 2025 de suelos heterogéneos de verano/arena/PCA).

## Correcciones realizadas al portar desde `pinnV3`

1. **`benchmarks/mms_layered_y`**: se corrigió la transposición de ejes entre la conductividad/fuente (`y <= 0.5`, `sin(pi*x)`) y la solución analítica declarada (`reference.exact`), verificando numéricamente la consistencia exacta de la EDP $-\nabla\cdot(k\nabla T) = Q$.
2. **`thesis_curated/xlpe_discrete_layers`**: se restauró la estratificación del suelo (`k = 0.5 W/(m·K)` bajo `y = -1.0 m` y `k = 1.0 W/(m·K)` sobre `y = -1.0 m`) manteniendo todas las capas concéntricas del cable XLPE con prioridades consecutivas válidas.
3. **`benchmarks/tabla_heterogenea_demo`**: nuevo caso demostrativo autocontenido con tabla cartesiana 2D (`table` bilineal), tabla 1D lineal, reglas lógicas `piecewise` y expresiones analíticas sobre `(x, y)`.
