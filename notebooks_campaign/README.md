# Campaña Experimental Reproducible en Embudo — `PINN_V4`

Esta carpeta separada contiene los **6 cuadernos Jupyter** necesarios para reproducir, auditar y repetir íntegramente el flujo de campañas, *smoke tests*, cribado (*screening*) y clasificación aplicado en la versión `pinnV3`, pero ejecutado sobre el núcleo orientado a objetos de **`PINN_V4`** (`pinn_thermal`) con caché **FEniCSx P2 vía WSL2**.

## Flujo en Embudo (Reducción Progresiva de Modelos)

```mermaid
flowchart LR
    S0["00: Auditoría MMS y FEM WSL2 P2"] --> S1["01: Smoke de 10 Arquitecturas"]
    S1 --> S2["02: Screening 1200+50 y Poda"]
    S2 --> S3["03: Smoke Multicaso sobre Promovidos"]
    S3 --> S4["04: Campaña Adaptativa y Corridas Largas 8000+20"]
    S4 --> S5["05: Clasificación STRICT y Reporte Tesis"]
```

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

## Criterios Oficiales de Calificación

- **Umbral Estricto (`STRICT`, calificación oficial)**: simultáneamente:
  - `error_Tmax_abs_K < 0.05 K`
  - `nrmse_pct < 0.25 %`
  - `balance_pct < 0.5 %`
- **Umbral Relajado (`RELAXED`, promoción temprana a L-BFGS en Etapa 4A)**: `2 × STRICT` (`error_Tmax_abs_K < 0.10 K`, `nrmse_pct < 0.50 %`, `balance_pct < 1.0 %`).

Los resultados intermedios, referencias FEM cacheadas en WSL2 (`fem_references/`), la lista de modelos promovidos (`promoted_models.json`) y las tablas consolidadas se almacenan en `notebooks_campaign/campaign_workspace/` con escritura atómica reanudable ante interrupciones.
