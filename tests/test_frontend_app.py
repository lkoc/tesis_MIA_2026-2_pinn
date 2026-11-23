"""Pruebas automatizadas de arranque y navegación del frontend Streamlit."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_streamlit_frontend_boots_all_four_pages(root_dir: Path) -> None:
    st_testing = pytest.importorskip("streamlit.testing.v1")
    app_file = root_dir / "frontend" / "app.py"
    at = st_testing.AppTest.from_file(str(app_file), default_timeout=30)
    at.run()
    assert not at.exception, f"Excepción inicializando vista 1: {at.exception}"

    options = list(at.sidebar.radio[0].options)
    for option in [*options[1:], options[0]]:
        at.sidebar.radio[0].set_value(option).run()
        assert not at.exception, f"Excepción en página {option!r}: {at.exception}"
