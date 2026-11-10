"""
Modulo de Componentes de Interfaz Clinica - NEURO-CALC
Componentes reutilizables: tarjetas de metricas y encabezado de estacion PACS.
"""

from typing import Optional


def render_metric_card(
    title: str,
    value: str,
    unit: str = "",
    badge_text: Optional[str] = None,
    badge_type: str = "stable"
) -> str:
    """Genera el HTML para una tarjeta de KPI clinico."""
    badge_html = ""
    if badge_text:
        badge_class = f"badge-{badge_type}"
        badge_html = f'<div class="clinical-card-badge {badge_class}">{badge_text}</div>'

    return f"""
    <div class="clinical-card">
        <div class="clinical-card-header">{title}</div>
        <div class="clinical-card-value">{value}<span class="clinical-card-unit">{unit}</span></div>
        {badge_html}
    </div>
    """


def render_workstation_header(patient_id: str, modality: str, timestamp: str, lesion_type: str = "Lesion") -> str:
    """Genera el encabezado oficial de la estacion de trabajo diagnostica."""
    return f"""
    <div class="workstation-header">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
            <div>
                <div class="workstation-title">Estacion de Telemetria Tomografica PACS</div>
                <div class="workstation-subtitle">
                    CASO: <strong>{patient_id}</strong> &nbsp;|&nbsp;
                    MODALIDAD: <strong>{modality}</strong> &nbsp;|&nbsp;
                    PATOLOGIA: <strong>{lesion_type}</strong>
                </div>
            </div>
            <div style="text-align: right;">
                <span class="clinical-card-badge badge-stable">PIPELINE ACTIVO</span>
            </div>
        </div>
    </div>
    """
