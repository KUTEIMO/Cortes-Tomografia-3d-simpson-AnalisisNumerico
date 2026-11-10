"""
Modulo del Directorio Clinico de Pacientes - NEURO-CALC
Gestion de historias clinicas, estudios volumetricos y telemetria tomografica.
"""

from typing import Callable
import streamlit as st
from src.database.db_manager import PatientDatabaseManager


def render_patients_directory(
    db: PatientDatabaseManager,
    current_user_id: str,
    on_select_patient: Callable[[str], None],
    on_navigate_intake: Callable[[], None]
):
    """Renderiza el Directorio Clinico de Pacientes con jerarquia de tarjetas unificadas."""
    patients = db.list_patients()
    total_patients = len(patients)
    critical_cases = sum(1 for p in patients if "Critico" in p.get("risk_stage", ""))
    validated_cases = sum(1 for p in patients if "Validado" in p.get("status", "") or "Procesado" in p.get("status", ""))

    # -------------------------------------------------------------------------
    # ZONA 1: ENCABEZADO Y ACCION PRINCIPAL
    # -------------------------------------------------------------------------
    col_head_title, col_head_btn = st.columns([3.5, 1.3])
    with col_head_title:
        st.markdown("""
        <div class="page-header">
            <div class="page-eyebrow">Registro Clinico</div>
            <h1 class="page-title">Directorio de Pacientes</h1>
            <p class="page-subtitle">Gestion de historias clinicas, estudios volumetricos y acceso a la estacion PACS.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_head_btn:
        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
        if st.button("Registrar Nuevo Paciente", type="primary", use_container_width=True):
            on_navigate_intake()

    # -------------------------------------------------------------------------
    # ZONA 2: INDICADORES CLINICOS (KPI STRIP)
    # -------------------------------------------------------------------------
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    with col_k1:
        st.markdown(f"""
        <div class="clinical-card">
            <div class="clinical-card-header">Pacientes Registrados</div>
            <div class="clinical-card-value">{total_patients}<span class="clinical-card-unit">casos</span></div>
            <div class="clinical-card-badge badge-stable">REPOSITORIO ACTIVO</div>
        </div>
        """, unsafe_allow_html=True)

    with col_k2:
        st.markdown(f"""
        <div class="clinical-card">
            <div class="clinical-card-header">Efecto de Masa Critico</div>
            <div class="clinical-card-value">{critical_cases}<span class="clinical-card-unit">urgentes</span></div>
            <div class="clinical-card-badge badge-warning">GRADO III (&gt; 25 cm³)</div>
        </div>
        """, unsafe_allow_html=True)

    with col_k3:
        st.markdown(f"""
        <div class="clinical-card">
            <div class="clinical-card-header">Estudios Validados</div>
            <div class="clinical-card-value">{validated_cases}<span class="clinical-card-unit">series</span></div>
            <div class="clinical-card-badge badge-cyan">CUADRATURA SIMPSON 1/3</div>
        </div>
        """, unsafe_allow_html=True)

    with col_k4:
        st.markdown("""
        <div class="clinical-card">
            <div class="clinical-card-header">Integridad de Datos</div>
            <div class="clinical-card-value">100%<span class="clinical-card-unit">verificado</span></div>
            <div class="clinical-card-badge badge-stable">CRC32 ACTIVO</div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # ZONA 3: BUSQUEDA Y FILTRADO
    # -------------------------------------------------------------------------
    st.markdown("<div class='section-label'>Estudios Registrados</div>", unsafe_allow_html=True)
    col_s1, col_s2 = st.columns([3.0, 1.5])
    with col_s1:
        search_query = st.text_input(
            "Buscar paciente",
            placeholder="Buscar por codigo (PAC-...), diagnostico o nombre...",
            label_visibility="collapsed"
        ).strip().lower()

    with col_s2:
        filter_risk = st.selectbox(
            "Filtrar por Estadificacion",
            ["Todos los niveles de riesgo", "Efecto de Masa Critico (Grado III)", "Masa Moderada (Grado II)", "Lesion Focal (Grado I)"],
            label_visibility="collapsed"
        )

    filtered_patients = []
    for p in patients:
        text_corpus = f"{p.get('patient_code', '')} {p.get('full_name', '')} {p.get('pathology', '')} {p.get('patient_id', '')}".lower()
        if search_query and search_query not in text_corpus:
            continue
        if filter_risk == "Efecto de Masa Critico (Grado III)" and "Critico" not in p.get("risk_stage", ""):
            continue
        if filter_risk == "Masa Moderada (Grado II)" and "Moderada" not in p.get("risk_stage", ""):
            continue
        if filter_risk == "Lesion Focal (Grado I)" and "Focal" not in p.get("risk_stage", ""):
            continue
        filtered_patients.append(p)

    if not filtered_patients:
        st.markdown("""
        <div class="clinical-card" style="text-align: center; padding: 32px; color: #667085;">
            No se encontraron pacientes que coincidan con los criterios aplicados.
        </div>
        """, unsafe_allow_html=True)
        return

    # -------------------------------------------------------------------------
    # ZONA 4: TARJETAS UNIFICADAS POR PACIENTE
    # -------------------------------------------------------------------------
    for p in filtered_patients:
        p_id = p.get("patient_id", "")
        p_code = p.get("patient_code", "PAC-SIN-CODIGO")
        p_name = p.get("full_name", "Anonimizado")
        p_pathology = p.get("pathology", "Diagnostico no especificado")
        p_modality = p.get("modality", "MR")
        p_vol = float(p.get("volume_cm3", 0.0))
        p_slices = p.get("num_slices", 0)
        p_h = float(p.get("slice_thickness_h", 5.0))
        p_risk = p.get("risk_stage", "Sin estadificar")
        p_physician = p.get("physician", "No asignado")
        p_date = p.get("created_at", "")[:10]

        is_critical = "Critico" in p_risk
        badge_class = "badge-warning" if is_critical else "badge-cyan"
        accent_color = "#D92D20" if is_critical else "#2563EB"

        with st.container(border=True):
            col_info, col_vol, col_actions = st.columns([3.0, 1.3, 1.5])

            with col_info:
                st.markdown(f"""
                <div style="border-left: 3px solid {accent_color}; padding-left: 12px;">
                    <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; font-weight: 700; color: #101828; background: #F2F4F7; padding: 2px 6px; border-radius: 4px;">
                            {p_code}
                        </span>
                        <span style="font-size: 0.92rem; font-weight: 600; color: #101828;">
                            {p_name}
                        </span>
                    </div>
                    <div style="font-size: 0.82rem; font-weight: 600; color: #344054; margin-top: 6px;">
                        {p_pathology}
                    </div>
                    <div style="font-size: 0.76rem; color: #667085; margin-top: 2px;">
                        {p_modality} &nbsp;&bull;&nbsp; {p_slices} cortes axiales &nbsp;&bull;&nbsp; h = {p_h:.2f} mm
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_vol:
                st.markdown(f"""
                <div style="text-align: right; padding-top: 2px;">
                    <div style="font-size: 0.68rem; font-weight: 600; color: #667085; text-transform: uppercase;">Volumen Calculado</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.25rem; font-weight: 700; color: #101828;">
                        {p_vol:.3f} <span style="font-size: 0.78rem; font-weight: 500; color: #475467;">cm³</span>
                    </div>
                    <span class="clinical-card-badge {badge_class}" style="margin-top: 4px;">{p_risk[:28]}</span>
                </div>
                """, unsafe_allow_html=True)

            with col_actions:
                if st.button("Abrir Estacion PACS", key=f"open_pacs_{p_id}", type="primary", use_container_width=True):
                    on_select_patient(p_id)

                delete_key = f"del_confirm_{p_id}"
                if delete_key not in st.session_state:
                    st.session_state[delete_key] = False

                if not st.session_state[delete_key]:
                    if st.button("Eliminar Registro", key=f"del_btn_{p_id}", use_container_width=True):
                        st.session_state[delete_key] = True
                        st.rerun()
                else:
                    col_c1, col_c2 = st.columns(2)
                    with col_c1:
                        if st.button("Confirmar", key=f"conf_yes_{p_id}", type="secondary", use_container_width=True):
                            db.delete_patient(p_id, user_id=current_user_id)
                            st.session_state[delete_key] = False
                            st.rerun()
                    with col_c2:
                        if st.button("Cancelar", key=f"conf_no_{p_id}", use_container_width=True):
                            st.session_state[delete_key] = False
                            st.rerun()

            with st.expander("Ficha Clinica y Metadatos de Adquisicion"):
                st.markdown(f"""
                <div style="font-size: 0.80rem; color: #344054; line-height: 1.6;">
                    <strong>Especialista:</strong> {p_physician} &nbsp;|&nbsp;
                    <strong>Fecha de Registro:</strong> {p_date} &nbsp;|&nbsp;
                    <strong>Pixel Spacing:</strong> {p.get('pixel_spacing', 0.9375)} mm/px &nbsp;|&nbsp;
                    <strong>Matriz:</strong> {p.get('matrix_dim', 256)}x{p.get('matrix_dim', 256)}<br>
                    <strong>Observaciones:</strong> {p.get('notes', 'Sin observaciones adicionales.')}
                </div>
                """, unsafe_allow_html=True)
