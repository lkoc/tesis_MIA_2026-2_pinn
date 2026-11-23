# Compatibilidad entre Modelos, Transferencia, Continuación y Encadenamiento

Uno de los objetivos centrales de `PINN_V4` es garantizar que ninguna solución quede aislada ni dependa de cómo o dónde fue entrenada (Jupyter, Streamlit o CLI) ni del modelo exacto empleado.

## 1. Formato Único de Artefacto (`SolutionArtifact`, `pinn-v4-artifact-1`)

Toda corrida guardada contiene:
- `problem.json`: especificación completa y autocontenida del problema.
- `specification.json`: arquitectura, hiperparámetros, semillas, dispositivo y `compatibility_profile`.
- `model.pt`: pesos finales, mejor estado de entrenamiento, estado del optimizador/scheduler/RNG/atención para reanudación.
- `result.json` y `evaluation.npz`: historial de pérdidas, auditoría de balance energético y métricas.
- En corridas FEM/RBF: `fem_l*.json` / `fem_l*.npz` o `field.npz` con interpolación por subregión respetando discontinuidades materiales.

Los cuadernos Jupyter y la UI Streamlit leen y escriben sobre `frontend/workspace/runs/`, por lo que una red entrenada en Jupyter puede inspeccionarse, reanudarse o transferirse inmediatamente en Streamlit (y viceversa).

---

## 2. Cuatro Estrategias de Continuación y Reutilización

### A. Reanudación Exacta (`resume_from`)
Reanuda la optimización exactamente donde se detuvo: mismo `physics_sha256`, misma arquitectura, estados de `Adam`/`L-BFGS`, `CosineAnnealingLR`, generadores aleatorios y nubes de colocación.

### B. Transferencia de Pesos entre Modelos Compatibles (`WeightTransfer`)
`CompatibilityChecker.check(source_artifact, target_model)` verifica las **características críticas** sin exigir que el archivo JSON sea idéntico:
- Misma partición (`global` vs `region`), misma formulación (`strong` vs `mixed`), misma `width`, `depth`, `activation`, `input_normalization` y `domain_shape`.
- Subredes emparejadas por nombre (`region:soil`, `region:conductor`) o por tipo funcional de región (`soil`, `conductor`, `material`) con dimensiones tensoriales coincidentes.

Permite, por ejemplo, entrenar un cable con conductividad base $k = 1.0$, modificar el JSON a $k(x, y)$ heterogéneo o cambiar la potencia del conductor, y continuar el entrenamiento aprovechando los pesos ya convergidos mediante `transfer_mode="weights"`.

### C. Transferencia Funcional Universal (`FunctionalTransfer`)
Cuando cambian la arquitectura, las dimensiones (`width`/`depth`) o la estructura de regiones (e.g. `M0 -> CUSPR`, `FBPINN -> M1_EXACT`, `M1 -> M2`, o desde una malla discreta **`FEM -> PINN`** / **`RBF-FD -> PINN`**), `FunctionalTransfer` muestrea el campo origen $\mathbf{x} \mapsto (T, \mathbf{q})$ en el dominio físico y proyecta al modelo destino mediante:
- `functional_readout`: mínimos cuadrados ordinarios sobre la última capa lineal (`0` iteraciones).
- `functional_readout_gls`: mínimos cuadrados generalizados ajustando simultáneamente valores y gradientes $\nabla T$.
- `functional_pretrain`: ajuste supervisado rápido (temperatura y flujo en modelos mixtos `M2`), aplicable a absolutamente **cualquier** par origen-destino.

Con `transfer_mode="auto"`, el entrenador intenta transferencia directa de pesos si son compatibles y recurre automáticamente a transferencia funcional si no lo son.

### D. Encadenamiento Activo (`ChainedPINN`)
Usa la salida diferenciable $T_{\text{prev}}(\mathbf{x})$ de una solución PINN previa como campo base sobre el cual un nuevo modelo aprende la corrección o refinamiento:
- `chain_mode="residual"`: $T(\mathbf{x}) = T_{\text{prev}}(\mathbf{x}) + \alpha\,\bigl(T_{\text{nuevo}}(\mathbf{x}) - T_{\text{ref}}\bigr)$.
- `chain_mode="feature"`: combina $T_{\text{prev}}$ normalizado y la red destino mediante una compuerta lineal entrenable.

Las derivadas respecto de $(x, y)$ se propagan íntegramente por `autograd` sobre el campo total para que todas las pérdidas físicas PDE actúen sobre la temperatura física real.
