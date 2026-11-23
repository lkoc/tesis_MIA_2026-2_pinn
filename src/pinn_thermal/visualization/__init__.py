"""Subsistema desacoplado de muestreo en grilla, visualización interactiva/estática y editor de casos."""

from pinn_thermal.visualization.geometry_editor import CaseEditorHelper, load_material_presets
from pinn_thermal.visualization.grids import FieldGrid2D, GridSampler
from pinn_thermal.visualization.plots import InteractivePlotter, StaticPlotter

__all__ = [
    "FieldGrid2D",
    "GridSampler",
    "InteractivePlotter",
    "StaticPlotter",
    "CaseEditorHelper",
    "load_material_presets",
]
