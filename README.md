# Proyecto de Tesis UNI — `PINN_V4` (`pinn-thermal`)

> **Proyecto de Tesis de Maestría — Universidad Nacional de Ingeniería (UNI)**  
> **Unidad académica:** Unidad de Posgrado de la Facultad de Ingeniería Industrial y de Sistemas (FIIS-UNI)  
> **Programa:** Maestría en Inteligencia Artificial  
> **Tema:** **DISEÑO Y EVALUACIÓN DE UNA RED NEURONAL INFORMADA POR FÍSICA PARA ESTIMAR LA TEMPERATURA Y LA AMPACIDAD DE CABLES ELÉCTRICOS ENTERRADOS EN ENTORNOS HETEROGÉNEOS.**  
> **Tesistas:** Luis Enrique Koc Góngora; Herbert Antonio Meléndez García.  
> **Asesor:** Samuel Oporto.

Este repositorio integra el desarrollo completo del proyecto de tesis de la **Universidad Nacional de Ingeniería (UNI)**: el **Plan de Tesis**, el **borrador del documento de tesis**, la **colección de artículos científicos y papers de referencia**, y el artefacto computacional **`PINN_V4` (`pinn-thermal`)** — una librería Python **autocontenida, instalable vía `pip` / PyPI**, construida sobre **PyTorch**, documentada íntegramente en español y diseñada bajo programación orientada a objetos (POO) por capas para modelar conducción térmica estacionaria y ampacidad en cables subterráneos y medios heterogéneos 2D.

---

## Documentos de Tesis, Plan y Carpeta de Artículos / Papers

| Componente | Ubicación en el repositorio | Contenido principal |
| :--- | :--- | :--- |
| **Plan de Tesis** | [`plan_tesis_final/`](plan_tesis_final/) | Documento fuente LaTeX ([`plan_tesis_cables_pinn.tex`](plan_tesis_final/plan_tesis_cables_pinn.tex)), versión compilada ([`plan_tesis_cables_pinn.pdf`](plan_tesis_final/plan_tesis_cables_pinn.pdf)), capítulos (`capitulos/`), referencias ([`referencias_plan_tesis.bib`](plan_tesis_final/referencias_plan_tesis.bib)), matriz de consistencia (`matriz_consistencia/`) y guías UNI (`guias/`). |
| **Borrador del Documento de Tesis** | [`documento_tesis_borrador/`](documento_tesis_borrador/) | Manuscrito principal en LaTeX ([`tesis.tex`](documento_tesis_borrador/tesis.tex)), borrador compilado ([`tesis.pdf`](documento_tesis_borrador/tesis.pdf)), capítulos (`capitulos/`), tablas e imágenes (`tablas/`, `imagenes/`), referencias ([`referencias.bib`](documento_tesis_borrador/referencias.bib)), script de compilación ([`compilar.ps1`](documento_tesis_borrador/compilar.ps1)), guía ([`README.md`](documento_tesis_borrador/README.md)) y seguimiento ([`PENDIENTES.md`](documento_tesis_borrador/PENDIENTES.md)). |
| **Artículos y Papers (Bibliografía PDF)** | [`documento_tesis_borrador/bibliografia/`](documento_tesis_borrador/bibliografia/) | Colección de **80 artículos científicos, papers, normas técnicas (IEC/CIGRE/ASTM) y libros en PDF** sobre PINNs, conducción térmica, heterogeneidad de suelos y ampacidad de cables subterráneos. |
| **Literatura Metodológica y Datos Fuente** | [`plan_tesis_final/literatura/`](plan_tesis_final/literatura/) y [`plan_tesis_final/datos_fuente/`](plan_tesis_final/datos_fuente/) | Papers base de *Design Science Research* (DSR) y redacción científica, junto con extractos de texto y figuras de apoyo. |
| **Plataforma Computacional y Campañas** | [`src/pinn_thermal/`](src/pinn_thermal/), [`cases/`](cases/), [`notebooks/`](notebooks/), [`notebooks_campaign/`](notebooks_campaign/), [`frontend/`](frontend/) | Código fuente de la librería `pinn-thermal`, catálogo de casos JSON, cuadernos de campaña experimental, aplicación Streamlit y pruebas unitarias ([`tests/`](tests/)). |

---

## Características Principales

1. **Núcleo extensible por capas (`src/pinn_thermal/core/`)**:
   - Capa geométrica y de campos **100 % libre de PyTorch** (`expressions`, `fields`, `shapes`, `schema`, `problem`), utilizable también dentro de entornos externos como FEniCSx.
   - Capa de física, entrenamiento, auditoría energética, ampacidad y artefactos portables sobre `BasePINNModel` y `ModelRegistry`.
2. **Modelos PINN especializados y registrados (`src/pinn_thermal/models/`)**:
   - Línea base y escalados: `M0`, `M0R`, `M1`, `M2`.
   - Enriquecidos con discontinuidades/cúspides: `CUSP`, `CUSPR`, `DCSNN`, `FBPINN`.
   - Restricciones analíticas exactas: `M1_EXACT`, `M0_EXACT_BC`, `M1_EXACT_BC`, `DCSNN_EXACT_BC`, `M0_EXACT`, `CUSP_EXACT`, `DCSNN_EXACT`, `FBPINN_EXACT`.
   - Inicializadores discretos sin entrenamiento previo: `RBFFDInitializer` (`rbf_readout`, `rbf_pretrain`) y `FEMInitializer`.
   - Adaptador opcional: `DeepXDE_PFNN`, `DeepXDE_FNN`.
3. **Autoconfiguración total desde JSON (`pinn-v4-problem-1`)**:
   - Cualquier modelo arma la física completa leyendo únicamente el archivo JSON del caso, admitiendo **constantes**, **expresiones simbólicas en `(x, y)`**, **reglas lógicas `piecewise`**, **tablas 1D/2D**, **estratos `layers`** y **parches `patch`**.
4. **Compatibilidad, continuación, transferencia y encadenamiento**:
   - Reanudación exacta (`resume_from`), transferencia de pesos entre redes compatibles aunque difiera el problema JSON (`WeightTransfer`), transferencia funcional universal entre arquitecturas o desde FEM/RBF (`FunctionalTransfer`), y encadenamiento directo (`ChainedPINN`).
5. **Aceleración hardware transparente y paralelismo opcional**:
   - `DeviceManager` detecta y selecciona transparentemente **CPU, NVIDIA CUDA, AMD ROCm, Intel XPU, NPU, Apple MPS y DirectML** simplemente cambiando la variante de `torch` instalada.
   - `ParallelRunner` ejecuta campañas multi-caso y multi-modelo en paralelo mediante procesos `spawn`.
6. **Visualización desacoplada, editor interactivo, Streamlit, Jupyter y CLI**:
   - Visor interactivo Plotly (zoom, lupa, inspección de propiedades y anotaciones editables), láminas estáticas Matplotlib, `CaseEditorHelper`, aplicación Streamlit (`frontend/`) y 5 cuadernos Jupyter (`notebooks/`) intercambiando artefactos a través del mismo espacio de trabajo (`frontend/workspace/`).
7. **Integración FEniCSx P2**:
   - Funciona directamente con el entorno `fenicsx` existente en WSL2 (`Ubuntu`) en Windows, o de forma nativa en Linux/macOS/Docker.

---

## Instalación Rápida

```bash
cd PINN_V4

# Instalación editable base (núcleo + modelos + visualización Matplotlib/Plotly)
pip install -e .

# Con interfaz web Streamlit y cuadernos Jupyter
pip install -e ".[ui]"

# Con herramientas de prueba y desarrollo
pip install -e ".[ui,dev]"
```

Para construir paquete distribuible estándar (wheel y sdist para PyPI):

```bash
pip install build
python -m build
```

> **Autocontención**: toda la carpeta `PINN_V4/` es independiente. Si se comprime y traslada a otra máquina, tras instalar sus dependencias funciona sin referenciar ninguna ruta externa del repositorio original.

---

## Inicio Rápido

### 1. En Python

```python
from pinn_thermal.core.trainer import PINNTrainer, TrainConfig
from pinn_thermal.visualization import GridSampler, StaticPlotter

# Carga directa desde el catálogo autocontenido o cualquier archivo .json
cfg = TrainConfig(adam_steps=800, lbfgs_steps=40, width=32, depth=3, device="auto")
art = PINNTrainer(cfg).train("mms_layered_y", "CUSPR", "mi_corrida_cuspr", overwrite=True)

print("Evaluación:", art.result["evaluation"])
grid = GridSampler.sample_artifact_grid(art, nx=150, ny=150)
StaticPlotter.plot_temperature_field(grid, art.problem, output_path="mi_corrida_cuspr/campo.png")
```

### 2. En Línea de Comandos (`pinn-thermal`)

```bash
pinn-thermal info
pinn-thermal list-models
pinn-thermal list-cases
pinn-thermal validate xlpe_discrete_layers
pinn-thermal train --problem xlpe_discrete_layers --arch M1_EXACT --out ./salidas/xlpe --focused --overwrite
pinn-thermal fem --problem xlpe_discrete_layers --out ./salidas/fem --level 2
pinn-thermal ui
```

### 3. Interfaz Gráfica Streamlit y Cuadernos Jupyter

```bash
# Aplicación web interactiva (editor visual + texto JSON, ejecución, comparador)
streamlit run frontend/app.py

# Cuadernos reproducibles paso a paso (flujo general)
jupyter lab notebooks/

# Cuadernos separados de la campaña experimental completa (screening -> corridas largas)
jupyter lab notebooks_campaign/
```

Consulta también [`notebooks/README.md`](notebooks/README.md) y [`notebooks_campaign/README.md`](notebooks_campaign/README.md) para ejecutar los *smoke tests*, cribado multi-criterio, caché FEM WSL2 y clasificación final `STRICT`.

Los casos creados en `frontend/workspace/user_cases/` y las soluciones guardadas en `frontend/workspace/runs/` mediante Jupyter o CLI aparecen automáticamente en el comparador de Streamlit (y viceversa).

---

## Documentación Detallada

- [`docs/arquitectura.md`](docs/arquitectura.md): diseño por capas, clases principales y flujo de datos.
- [`docs/teoria_modelos_pinn.md`](docs/teoria_modelos_pinn.md): formulación matemática, autores originales (con DOI) y aportes implementados.
- [`docs/formato_json.md`](docs/formato_json.md): manual completo para crear y editar casos 2D en texto o desde la interfaz.
- [`docs/transferencia_y_compatibilidad.md`](docs/transferencia_y_compatibilidad.md): reanudación, pesos compatibles, transferencia funcional y encadenamiento `ChainedPINN`.
- [`docs/aceleracion_y_paralelismo.md`](docs/aceleracion_y_paralelismo.md): uso de GPU/XPU/NPU/CPU y paralelismo multi-proceso.
- [`docs/instalacion_fenicsx.md`](docs/instalacion_fenicsx.md): configuración de FEniCSx vía WSL2 (estado actual), Linux nativo o Docker.
