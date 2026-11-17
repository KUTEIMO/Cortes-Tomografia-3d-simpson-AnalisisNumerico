"""
Modulo del Directorio Clinico de Pacientes - NEURO-CALC
"""

from typing import Callable
import streamlit as st
from src.database.db_manager import PatientDatabaseManager


def _format_risk_pill(risk_stage: str) -> tuple[str, str]:
    """Retorna etiqueta compacta y clase CSS sobria segun estadificacion clinica."""
    if "Critico" in risk_stage or "Grado III" in risk_stage:
        return ("Critico", "pill-critical")
    if "Moderada" in risk_stage or "Grado II" in risk_stage:
        return ("Moderado", "pill-warning")
    if "Focal" in risk_stage or "Grado I" in risk_stage:
        return ("Focal", "pill-stable")
    return (risk_stage or "Sin estadificar", "pill-info")


def render_patients_directory(
    db: PatientDatabaseManager,
    current_user_id: str,
    on_select_patient: Callable[[str], None],
    on_navigate_intake: Callable[[], None]
):
    """Renderiza el Directorio de Pacientes con jerarquia clinica estructurada."""
    patients = db.list_patients()
    total_patients = len(patients)
    critical_cases = sum(1 for p in patients if "Critico" in p.get("risk_stage", "") or "Grado III" in p.get("risk_stage", ""))
    validated_cases = sum(1 for p in patients if "Validado" in p.get("status", "") or "Procesado" in p.get("status", ""))

    # -------------------------------------------------------------------------
    # NIVEL 1: ENCABEZADO PRINCIPAL DE PAGINA Y CTA PRIMARIO
    # -------------------------------------------------------------------------
    col_head_title, col_head_btn = st.columns([3.8, 1.2])
    with col_head_title:
        st.markdown("""
        <div class="page-header-wrap">
            <div class="page-eyebrow">REGISTRO CLINICO / DIRECTORIO</div>
            <h1 class="page-title">Directorio de pacientes</h1>
            <p class="page-subtitle">Consulta y gestion de pacientes, estudios tomograficos y analisis volumetricos.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_head_btn:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("+ Registrar paciente", type="primary", width="stretch", key="btn_new_patient_dir"):
            on_navigate_intake()

    # -------------------------------------------------------------------------
    # NIVEL 2: INDICADORES OPERATIVOS SEPARADOS DE ALERTAS CLINICAS
    # -------------------------------------------------------------------------
    col_k1, col_k2, col_k3 = st.columns(3)
    with col_k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Pacientes</div>
            <div class="kpi-value">{total_patients:02d}</div>
            <div class="kpi-context">Historias clinicas activas</div>
        </div>
        """, unsafe_allow_html=True)

    with col_k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label" style="color: #991B1B !important;">Alertas criticas</div>
            <div class="kpi-value" style="color: #DC2626 !important;">{critical_cases:02d}</div>
            <div class="kpi-context" style="color: #991B1B !important;">Efecto de masa Grado III</div>
        </div>
        """, unsafe_allow_html=True)

    with col_k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Estudios validados</div>
            <div class="kpi-value">{validated_cases:02d}</div>
            <div class="kpi-context">Volumetria verificada</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # NIVEL 3: SUPERFICIE PRINCIPAL DE ESTUDIOS REGISTRADOS Y BUSQUEDA
    # -------------------------------------------------------------------------
    with st.container(border=True):
        st.markdown("<div class='section-heading'>Estudios registrados</div>", unsafe_allow_html=True)

        col_s1, col_s2 = st.columns([3.4, 1.4])
        with col_s1:
            search_query = st.text_input(
                "Buscar estudio",
                placeholder="Buscar por paciente, codigo o diagnostico...",
                label_visibility="collapsed"
            ).strip().lower()

        with col_s2:
            filter_risk = st.selectbox(
                "Filtrar por estado",
                ["Todos los estados", "Critico (Grado III)", "Moderado (Grado II)", "Focal (Grado I)"],
                label_visibility="collapsed"
            )

        filtered_patients = []
        for p in patients:
            text_corpus = f"{p.get('patient_code', '')} {p.get('full_name', '')} {p.get('pathology', '')} {p.get('patient_id', '')}".lower()
            if search_query and search_query not in text_corpus:
                continue
            if filter_risk == "Critico (Grado III)" and "Critico" not in p.get("risk_stage", "") and "Grado III" not in p.get("risk_stage", ""):
                continue
            if filter_risk == "Moderado (Grado II)" and "Moderada" not in p.get("risk_stage", "") and "Grado II" not in p.get("risk_stage", ""):
                continue
            if filter_risk == "Focal (Grado I)" and "Focal" not in p.get("risk_stage", "") and "Grado I" not in p.get("risk_stage", ""):
                continue
            filtered_patients.append(p)

        if not filtered_patients:
            st.markdown("""
            <div style="text-align: center; padding: 32px 16px; color: #64748B; font-size: 0.90rem;">
                No se encontraron pacientes que coincidan con el criterio de busqueda.
            </div>
            """, unsafe_allow_html=True)
            return

        st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

        for p in filtered_patients:
            p_id = p.get("patient_id", "")
            p_code = p.get("patient_code", "PAC-SIN-CODIGO")
            p_name = p.get("full_name", "Paciente anonimizado")
            p_pathology = p.get("pathology", "Diagnostico no especificado")
            p_modality = p.get("modality", "RM")
            p_vol = float(p.get("volume_cm3", 0.0))
            p_slices = p.get("num_slices", 0)
            p_h = float(p.get("slice_thickness_h", 5.0))
            p_risk = p.get("risk_stage", "")
            p_physician = p.get("physician", "No asignado")
            p_date = p.get("created_at", "")[:10]

            pill_label, pill_cls = _format_risk_pill(p_risk)

            with st.container(border=True):
                # Encabezado del paciente: Codigo e indicador de alerta
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.95rem; font-weight: 700; color: #0F172A;">
                        {p_code}
                    </span>
                    <span class="status-pill {pill_cls}">{pill_label}</span>
                </div>
                <div style="font-size: 1.05rem; font-weight: 600; color: #0F172A; margin-bottom: 2px;">
                    {p_name}
                </div>
                <div style="font-size: 0.85rem; color: #64748B; padding-bottom: 12px; border-bottom: 1px solid #F1F5F9; margin-bottom: 12px;">
                    {p_pathology} &middot; {p_modality} axial FLAIR
                </div>
                """, unsafe_allow_html=True)

                # Metricas agrupadas de estudio: Volumen y Adquisicion
                col_prop1, col_prop2 = st.columns(2)
                with col_prop1:
                    st.markdown(f"""
                    <div>
                        <div style="font-size: 0.76rem; color: #64748B; margin-bottom: 2px;">Volumen calculado</div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.25rem; font-weight: 700; color: #0F172A;">
                            {p_vol:.3f} cm³
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_prop2:
                    st.markdown(f"""
                    <div style="text-align: right;">
                        <div style="font-size: 0.76rem; color: #64748B; margin-bottom: 2px;">Adquisicion</div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.02rem; font-weight: 600; color: #0F172A;">
                            {p_slices} cortes &middot; {p_h:.0f} mm
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

                # CTA Principal a ancho completo
                if st.button("Abrir estudio en PACS", key=f"open_pacs_{p_id}", type="primary", width="stretch"):
                    on_select_patient(p_id)

                # Ficha clinica y acciones secundarias contextualizadas
                with st.expander("Ver ficha clinica y metadatos →"):
                    st.markdown(f"""
                    <div style="font-size: 0.82rem; color: #334155; line-height: 1.65; margin-bottom: 10px;">
                        <strong>Especialista asignado:</strong> {p_physician} &nbsp;|&nbsp;
                        <strong>Fecha de registro:</strong> {p_date}<br>
                        <strong>Calibracion espacial:</strong> {p.get('pixel_spacing', 0.9375)} mm/px &nbsp;|&nbsp;
                        <strong>Matriz:</strong> {p.get('matrix_dim', 256)}x{p.get('matrix_dim', 256)}<br>
                        <strong>Notas clinicas:</strong> {p.get('notes', 'Sin observaciones adicionales.')}
                    </div>
                    """, unsafe_allow_html=True)

                    delete_key = f"del_confirm_{p_id}"
                    if delete_key not in st.session_state:
                        st.session_state[delete_key] = False

                    if not st.session_state[delete_key]:
                        if st.button("Eliminar registro de paciente", key=f"del_btn_{p_id}", type="secondary"):
                            st.session_state[delete_key] = True
                            st.rerun()
                    else:
                        st.markdown("<div style='font-size: 0.78rem; color: #991B1B; margin-bottom: 6px;'>¿Confirmar baja definitiva del paciente y sus series tomograficas?</div>", unsafe_allow_html=True)
                        col_c1, col_c2 = st.columns(2)
                        with col_c1:
                            if st.button("Confirmar baja", key=f"conf_yes_{p_id}", type="primary", width="stretch"):
                                db.delete_patient(p_id, user_id=current_user_id)
                                st.session_state[delete_key] = False
                                st.rerun()
                        with col_c2:
                            if st.button("Cancelar", key=f"conf_no_{p_id}", width="stretch"):
                                st.session_state[delete_key] = False
                                st.rerun()
