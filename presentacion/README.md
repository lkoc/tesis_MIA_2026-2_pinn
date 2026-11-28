# Presentación Técnica y Experimental — `PINN_V4` (`pinn_thermal`)

Esta carpeta contiene la presentación LaTeX **Beamer (16:9)** con la arquitectura del software, detalles de los motores PINN y FEM, formato JSON, uso del front-end Streamlit, catálogo detallado de modelos PINN y los resultados experimentales obtenidos en la campaña.

## Contenido

- [`presentacion_pinn_v4.tex`](presentacion_pinn_v4.tex): fuente LaTeX Beamer con diagramas TikZ, ecuaciones, tablas y enlaces relativos al código y cuadernos del repositorio.
- [`presentacion_pinn_v4.pdf`](presentacion_pinn_v4.pdf): presentación compilada lista para exponer.
- [`launch_frontend.bat`](launch_frontend.bat): lanzador directo en Windows para arrancar Streamlit (`frontend/app.py`) si aún no está activo y abrir `http://localhost:8501` en el navegador (enlazado desde la diapositiva del Front-End).
- [`generar_figuras.py`](generar_figuras.py): script reproducible que lee los JSON reales de `notebooks_campaign/campaign_workspace/` y genera los gráficos vectoriales en `figuras/`.

## Recompilación rápida

Desde la raíz del repositorio:

```powershell
python presentacion\generar_figuras.py
cd presentacion
pdflatex -interaction=nonstopmode -halt-on-error presentacion_pinn_v4.tex
pdflatex -interaction=nonstopmode -halt-on-error presentacion_pinn_v4.tex
```

> Nota: los enlaces relativos (`../src/...`, `../cases/...`, `../notebooks_campaign/...`, `../frontend/...`, `launch_frontend.bat`) y los enlaces al Front-End (`http://localhost:8501`) funcionan abriendo `presentacion_pinn_v4.pdf` desde un visor PDF local de escritorio (Adobe Reader, SumatraPDF, Edge/Chrome).
