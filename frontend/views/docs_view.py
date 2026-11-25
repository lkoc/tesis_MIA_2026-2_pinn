"""Vista 4: Documentación integrada de modelos PINN, autores, modificaciones y manual JSON."""

from __future__ import annotations

from pathlib import Path
import streamlit as st  # type: ignore

import pinn_thermal.models  # noqa: F401
from pinn_thermal.core.base_model import ModelRegistry


def render_docs_view(root_dir: Path) -> None:
    st.header("4. Documentación Teórica, Autores y Manual del Formato JSON")

    t1, t2, t3, t4 = st.tabs(
        [
            "📚 Catálogo de Modelos Registrados",
            "📐 Fundamento Teórico",
            "🔬 Detalle Técnico y Auditoría PINN",
            "📄 Manual de Archivos JSON",
        ]
    )

    with t1:
        st.dataframe(ModelRegistry.describe_all(), use_container_width=True)

    with t2:
        theory_file = root_dir / "docs" / "teoria_modelos_pinn.md"
        if theory_file.is_file():
            st.markdown(theory_file.read_text(encoding="utf-8-sig"))
        else:
            st.info("Consulte `docs/teoria_modelos_pinn.md` una vez generada la documentación.")

    with t3:
        detail_file = root_dir / "docs" / "detalle_tecnico_implementaciones_pinn.md"
        if detail_file.is_file():
            st.markdown(detail_file.read_text(encoding="utf-8-sig"))
        else:
            st.info("Consulte `docs/detalle_tecnico_implementaciones_pinn.md` una vez generada la documentación.")

    with t4:
        json_file = root_dir / "docs" / "formato_json.md"
        if json_file.is_file():
            st.markdown(json_file.read_text(encoding="utf-8-sig"))
        else:
            st.info("Consulte `docs/formato_json.md`.")
