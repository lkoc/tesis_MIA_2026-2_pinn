"""Vistas modulares de la interfaz gráfica Streamlit para ``PINN_V4``."""

from frontend.views.case_studio import render_case_studio
from frontend.views.compare_view import render_compare_view
from frontend.views.docs_view import render_docs_view
from frontend.views.runner_view import render_runner_view

__all__ = [
    "render_case_studio",
    "render_runner_view",
    "render_compare_view",
    "render_docs_view",
]
