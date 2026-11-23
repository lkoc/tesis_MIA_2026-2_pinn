# Cuadernos Jupyter Reproducibles — `PINN_V4`

Estos cinco cuadernos interactivos siguen el flujo completo de trabajo importando directamente la librería `pinn_thermal` y compartiendo casos/artefactos con la interfaz Streamlit a través de `frontend/workspace/`.

1. **`01_introduccion_y_formato_json.ipynb`**: exploración del catálogo, construcción guiada de problemas heterogéneos (`table` 1D/2D, `piecewise`, expresiones `(x, y)`), anotaciones y visor Plotly.
2. **`02_entrenamiento_modelos_pinn_y_fem.ipynb`**: entrenamiento comparativo (`M1`, `CUSPR`), detección automática de aceleradores (`CPU`/`CUDA`/`XPU`) e integración FEniCSx WSL2/nativo.
3. **`03_transferencia_continuacion_y_encadenamiento.ipynb`**: reanudación exacta, transferencia de pesos entre JSON distintos, transferencia funcional entre arquitecturas distintas y encadenamiento `ChainedPINN`.
4. **`04_casos_curados_tesis_y_ampacidad.ipynb`**: cables XLPE multicapa en suelos estratificados (`xlpe_discrete_layers`), inicialización RBF-FD y cálculo de ampacidad admisible.
5. **`05_paralelismo_y_campana_multicaso.ipynb`**: campañas multi-caso y multi-modelo en paralelo mediante `ParallelRunner`.
