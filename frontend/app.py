"""Aplicación web interactiva Streamlit para ``PINN_V4`` (``pinn-thermal``).

Ejecutar con::

    streamlit run frontend/app.py
"""

from __future__ import annotations

from pathlib import Path
import sys

# Garantiza que ``PINN_V4/src`` y ``PINN_V4/frontend`` sean importables aun sin pip install
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for candidate in (SRC_DIR, ROOT_DIR):
    text = str(candidate)
    if text not in sys.path:
        sys.path.insert(0, text)

import streamlit as st  # type: ignore

from pinn_thermal.core.device import DeviceManager
from pinn_thermal.fem.runner import FEniCSxAdapter
from frontend.state import WorkspaceStore
from frontend.views import render_case_studio, render_compare_view, render_docs_view, render_runner_view


def main() -> None:
    st.set_page_config(
        page_title="PINN_V4 — Laboratorio Térmico 2D de Cables Subterráneos",
        page_icon="🔥",
        layout="wide",
    )

    store = WorkspaceStore(ROOT_DIR / "frontend" / "workspace", ROOT_DIR / "cases")
    hw = DeviceManager.describe()
    fem_status = FEniCSxAdapter().is_available()

    with st.sidebar:
        st.title("🔥 PINN_V4")
        st.caption("Plataforma modular e intercambiable para conducción térmica 2D en cables heterogéneos.")
        page = st.radio(
            "Navegación",
            [
                "1. Editor de Casos y Geometría",
                "2. Entrenamiento, Transferencia y FEM",
                "3. Visor y Comparación de Resultados",
                "4. Teoría, Autores y Manual JSON",
            ],
        )
        st.divider()
        st.subheader("Entorno de Ejecución")
        avail_backends = [name for name, info in hw["backends"].items() if info.get("available")]
        st.markdown(f"- **Hardware activo**: `{hw['recommended_device']}`")
        st.markdown(f"- **Backends disponibles**: `{', '.join(avail_backends)}`")
        if fem_status.get("available"):
            st.success(f"FEniCSx listo ({fem_status.get('backend')})")
        else:
            st.warning("FEniCSx no detectado en el backend por defecto")
        st.caption(f"Espacio compartido con Jupyter:\n`{store.root}`")

    if page.startswith("1."):
        render_case_studio(store)
    elif page.startswith("2."):
        render_runner_view(store)
    elif page.startswith("3."):
        render_compare_view(store)
    else:
        render_docs_view(ROOT_DIR)


if __name__ == "__main__":
    main()
