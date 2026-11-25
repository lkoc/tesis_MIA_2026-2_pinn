# Interfaz Web Streamlit — `PINN_V4`

Inspirada en el flujo visual de `PINN_v2` y construida enteramente sobre `pinn_thermal` (Capa 1 + Capa 2 + Capa 3).

## Características principales

1. **Estudio de Casos 2D (`case_studio.py`)**: creación guiada y por texto JSON, biblioteca de materiales autocontenida, editor visual de heterogeneidades (`constante`, `expresión (x, y)`, `piecewise`, `tabla 1D/2D`, `layers`, `patch`), anotaciones y visor interactivo Plotly con zoom y dibujo.
2. **Ejecutor de Modelos (`runner_view.py`)**: entrenamiento directo, reanudación exacta, transferencia de pesos o transferencia funcional, encadenamiento residual/feature (`ChainedPINN`), inicialización RBF-FD / FEM, ejecución FEniCSx (WSL2/nativo/Docker) y cálculo de ampacidad.
3. **Comparador Universal (`compare_view.py`)**: visualiza y contrasta sin distinción corridas creadas desde **Streamlit**, **Jupyter** o **CLI** almacenadas en `frontend/workspace/runs/`.
4. **Teoría, Auditoría Técnica y Manual JSON (`docs_view.py`)**: integra en vivo el catálogo `ModelRegistry`, los fundamentos teóricos (`docs/teoria_modelos_pinn.md`), la auditoría técnica frente a papers fundacionales (`docs/detalle_tecnico_implementaciones_pinn.md`) y el manual del esquema JSON (`docs/formato_json.md`).

## Lanzamiento

```bash
cd PINN_V4
streamlit run frontend/app.py
# O mediante el comando CLI del paquete instalado:
pinn-thermal ui
```
