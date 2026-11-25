# Campaña Experimental Reproducible en Embudo — `PINN_V4`

Esta carpeta separada contiene los **6 cuadernos Jupyter** necesarios para reproducir, auditar y repetir íntegramente el flujo de campañas, *smoke tests*, cribado (*screening*) y clasificación aplicado en la versión `pinnV3`, pero ejecutado sobre el núcleo orientado a objetos de **`PINN_V4`** (`pinn_thermal`) con caché **FEniCSx P2 vía WSL2**.

## Flujo en Embudo (Reducción Progresiva de Modelos)

```mermaid
flowchart LR
    S0["00: Auditoría MMS y FEM WSL2 P2"] --> S1["01: Smoke de 10 Arquitecturas"]
    S1 --> S2["02: Screening Dual (with/no energy) y Poda Acotada"]
    S2 --> S3["03: Smoke Multicaso sobre Promovidos"]
    S3 --> S4["04: Campaña Adaptativa y Corridas Largas 8000+20"]
    S4 --> S5["05: Clasificación STRICT y Reporte Tesis"]
```

> **Criterio de Etapa 2 (`02_screening_seleccion_modelos_1200_50.ipynb`)**: evalúa todas las arquitecturas en dos pasadas (`with_energy` con `"energy": 100.0` y `no_energy` con `"energy": 0.0`) manteniendo activo `interface_flux=100.0`, con `divergence_stop` temprano, puntuación compuesta logarítmica `log1p` y compuerta de sanidad sobre `mms_sine` antes de promover hasta 5 modelos a las Etapas 3 y 4.

## Presupuestos por Modo de Ejecución (`RUN_MODE`)

Cada cuaderno declara al inicio la variable `RUN_MODE`:

| Parámetro | `"demo_fast"` (validación rápida) | `"full_campaign"` (tesis completa) |
| :--- | :--- | :--- |
| Casos evaluados en Etapas 0, 3, 4 | Subconjunto representativo (`2–4` casos) | Catálogo completo (`29` casos) |
| Modelos en Etapa 1 / 2 | Subconjunto base (`5–6` modelos) | Abanico completo (`10` arquitecturas) |
| Presupuesto Screening (Etapa 2) | `120 Adam + 15 L-BFGS` | `1200 Adam + 50 L-BFGS` |
| Semillas multi-corrida (Etapa 3 y 4A) | `(0,)` | `(0, 1, 2)` |
| Bloques Adaptativos (Etapa 4A) | `ADAM=(60, 60)`, `LBFGS=(10, 10)` | `ADAM=(500, 500, 500)`, `LBFGS=(50, 50, 50, 50)` |
| Corrida Larga Profunda (Etapa 4B) | `120 Adam + 15 L-BFGS` | `8000 Adam + 20 L-BFGS` con `AccuracyStop` |

## Paralelismo y Selección de Dispositivo (`USE_PARALLEL`)

Los cuadernos de entrenamiento (`01` a `04`) incorporan las banderas `USE_PARALLEL`, `MAX_WORKERS` y `DEVICE`:

- **`USE_PARALLEL = True` (predeterminado)**: fija automáticamente `DEVICE = "cpu"` y ejecuta las corridas en paralelo con `ParallelRunner` (contexto `spawn`, reparto automático de hilos CPU por proceso).
- **`USE_PARALLEL = False`**: fija `DEVICE = "auto"` y ejecuta secuencialmente en el acelerador detectado (p. ej. Intel XPU con parche nativo de `aten::dot` en `DeviceManager.patch_xpu_ops()`, sin fallbacks a CPU).
- **Avance en vivo (`show_progress = True`)**: `ParallelRunner` muestra en los cuadernos un panel HTML actualizado en sitio (o resumen en consola) con barra global por pasos ejecutados, porcentaje, fase (`adam` / `lbfgs`), `loss` actual, tiempo transcurrido y ETA estimada leyendo `progress.json` de cada trabajador.

## Criterios Oficiales de Calificación

- **Umbral Estricto (`STRICT`, calificación oficial)**: simultáneamente:
  - `error_Tmax_abs_K < 0.05 K`
  - `nrmse_pct < 0.25 %`
  - `balance_pct < 0.5 %`
- **Umbral Relajado (`RELAXED`, promoción temprana a L-BFGS en Etapa 4A)**: `2 × STRICT` (`error_Tmax_abs_K < 0.10 K`, `nrmse_pct < 0.50 %`, `balance_pct < 1.0 %`).

Los resultados intermedios, referencias FEM cacheadas en WSL2 (`fem_references/`), la lista de modelos promovidos (`promoted_models.json`) y las tablas consolidadas se almacenan en `notebooks_campaign/campaign_workspace/` con escritura atómica reanudable ante interrupciones.
