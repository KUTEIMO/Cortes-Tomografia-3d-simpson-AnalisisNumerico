"""
Paquete del Modulo de Interfaz de Usuario de NeuroCalc
"""
from src.ui.styles import CLINICAL_CSS
from src.ui.components import render_metric_card, render_workstation_header
from src.ui.plotter_3d import (
    create_3d_full_brain_visualization,
    create_2d_area_profile,
    create_error_comparison_chart,
    create_2d_slice_view
)

__all__ = [
    "CLINICAL_CSS",
    "render_metric_card",
    "render_workstation_header",
    "create_3d_full_brain_visualization",
    "create_2d_area_profile",
    "create_error_comparison_chart",
    "create_2d_slice_view"
]

