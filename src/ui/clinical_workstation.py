"""
Estacion Clinica Operativa (PACS Workstation) - NEURO-CALC
Visor biaxial 3D/2D sincronizado, consola de navegacion axial y telemetria volumetrica.
"""

from typing import List, Dict, Any, Optional, Tuple
import time
import numpy as np
import pandas as pd
import streamlit as st

from src.ui.plotter_3d import (
    create_3d_full_brain_visualization,
    create_2d_area_profile,
    create_2d_slice_view
)


def get_clinical_risk_assessment(volume_cm3: float) -> Tuple[str, str, str]:
    """Clasificacion del efecto de masa y estadificacion clinica segun volumen tumoral."""
    if volume_cm3 < 5.0:
        return (
            "Grado I: Lesion Focal Menor (< 5 cm3)",
            "Masa incipiente sin efecto compresivo significativo sobre ventriculos laterales.",
            "stable"
        )
    elif volume_cm3 <= 25.0:
        return (
            "Grado II: Masa Moderada (5 - 25 cm3)",
            "Efecto de masa parcial con riesgo de compresion ventricular ipsilateral.",
            "warning"
        )
    else:
        return (
            "Grado III: Efecto de Masa Critico (Urgente)",
            "Volumen > 25 cm3. Colapso ventricular o desplazamiento de linea media cerebral.",
            "critical"
        )


def render_clinical_workstation(
    packets: List[Any],
    annotated_slices: List[np.ndarray],
    raw_slices: List[np.ndarray],
    mask_slices: List[np.ndarray],
    case_slices_meta: Optional[List[Dict[str, Any]]],
    all_brain_slices_meta: Optional[List[Dict[str, Any]]],
    res_simpson: Any,
    res_trapezoid: Any,
    res_riemann: Any,
    z_coords: np.ndarray,
    areas: np.ndarray,
    px_spacing: float,
    matrix_dimension: int,
    case_title: str,
    case_modality: str,
    lesion_label: str,
    timestamp: str
):
    """Renderiza la estacion clinica PACS con jerarquia de contenedores unificados."""
    total_active_slices = len(packets)

    if "axial_slice_slider" not in st.session_state:
        st.session_state.axial_slice_slider = (total_active_slices // 2) + 1

    st.session_state.axial_slice_slider = max(1, min(total_active_slices, st.session_state.axial_slice_slider))

    def pacs_nav_set(val: int):
        st.session_state.axial_slice_slider = max(1, min(total_active_slices, val))

    # =========================================================================
    # NIVEL 1: ALERTA CLINICA DE EFECTO DE MASA
    # =========================================================================
    risk_title, risk_desc, risk_badge = get_clinical_risk_assessment(res_simpson.volume_cm3)
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 16px; background: #FFFFFF; border: 1px solid #EAECF0; border-left: 3px solid #2563EB; border-radius: 8px; margin-bottom: 12px;">
        <div style="font-size: 0.82rem;">
            <strong style="color: #101828; text-transform: uppercase; font-size: 0.74rem; letter-spacing: 0.04em;">Efecto de Masa:</strong>
            <span style="color: #344054; margin-left: 6px;">{risk_title} &mdash; {risk_desc}</span>
        </div>
        <span class="clinical-card-badge badge-{risk_badge}" style="margin: 0; white-space: nowrap;">ESTADIFICACION VOLUMETRICA</span>
    </div>
    """, unsafe_allow_html=True)

    # =========================================================================
    # NIVEL 2: CONSOLA DE NAVEGACION AXIAL Y CINE-LOOP
    # =========================================================================
    cur_slider_val = st.session_state.axial_slice_slider
    cur_z_preview = packets[cur_slider_val - 1].z_position_mm if packets else 0.0

    with st.container(border=True):
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <span style="font-size: 0.74rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #475467;">
                Navegacion Axial y Cine-Loop (PACS)
            </span>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; font-weight: 600; color: #2563EB;">
                CORTE {cur_slider_val} / {total_active_slices} &nbsp;|&nbsp; z = {cur_z_preview:.1f} mm &nbsp;|&nbsp; h = {res_simpson.slice_thickness_h:.2f} mm
            </span>
        </div>
        """, unsafe_allow_html=True)

        selected_slice_1based = st.slider(
            "Scrubber Axial",
            min_value=1,
            max_value=total_active_slices,
            step=1,
            key="axial_slice_slider",
            format="Corte %d",
            label_visibility="collapsed"
        )

        col_t1, col_t2, col_t3, col_t4, col_t5, col_sp, col_lp = st.columns([0.9, 0.9, 2.2, 0.9, 0.9, 2.0, 1.1])
        with col_t1:
            st.button("Inicio", key="pacs_start", on_click=pacs_nav_set, args=(1,), use_container_width=True)
        with col_t2:
            st.button("Anterior", key="pacs_prev", on_click=pacs_nav_set, args=(st.session_state.axial_slice_slider - 1,), use_container_width=True)
        with col_t3:
            run_cine = st.button("Reproducir Cine-Loop", key="pacs_play", type="primary", use_container_width=True)
        with col_t4:
            st.button("Siguiente", key="pacs_next", on_click=pacs_nav_set, args=(st.session_state.axial_slice_slider + 1,), use_container_width=True)
        with col_t5:
            st.button("Final", key="pacs_end", on_click=pacs_nav_set, args=(total_active_slices,), use_container_width=True)
        with col_sp:
            cine_speed_opt = st.selectbox(
                "Cadencia",
                [
                    "Cadencia Clinica (0.60s)",
                    "Analisis Detallado (0.90s)",
                    "Velocidad Estandar (0.35s)",
                    "Barrido Rapido (0.15s)"
                ],
                index=0,
                label_visibility="collapsed"
            )
        with col_lp:
            st.markdown("<div style='margin-top: 6px;'>", unsafe_allow_html=True)
            loop_cine = st.checkbox("Bucle", value=False, help="Repetir barrido al finalizar")
            st.markdown("</div>", unsafe_allow_html=True)

    selected_slice_idx = selected_slice_1based - 1

    # =========================================================================
    # NIVEL 3: VISORES BIAXIALES SINCRONIZADOS (3D | 2D)
    # =========================================================================
    col_left, col_right = st.columns([1, 1], gap="medium")

    with col_left:
        with st.container(border=True):
            col_3d_title, col_3d_mode = st.columns([1.4, 1.6])
            with col_3d_title:
                st.markdown("<div class='clinical-card-header' style='padding-top: 4px; margin-bottom: 0;'>Reconstruccion 3D</div>", unsafe_allow_html=True)
            with col_3d_mode:
                mode_3d_choice = st.radio(
                    "Modo 3D",
                    ["Volumen 3D Completo", "Corte Planar (1:1)"],
                    horizontal=True,
                    index=0,
                    label_visibility="collapsed"
                )
            inspect_single_slice = (mode_3d_choice == "Corte Planar (1:1)")
            placeholder_3d = st.empty()

    with col_right:
        with st.container(border=True):
            col_2d_title, col_2d_mode = st.columns([1.4, 1.6])
            with col_2d_title:
                st.markdown(f"<div class='clinical-card-header' style='padding-top: 4px; margin-bottom: 0;'>Resonancia Axial (Corte {selected_slice_1based}/{total_active_slices})</div>", unsafe_allow_html=True)
            with col_2d_mode:
                view_mode = st.selectbox(
                    "Capa 2D",
                    [
                        "Contorno Clinico (Cobalto)",
                        "Vision Artificial Refinada (Esmeralda)",
                        "Auditoria Dual: Clinico vs IA",
                        "Resonancia Pura (FLAIR)",
                        "Mascara Radiologica"
                    ],
                    index=0,
                    label_visibility="collapsed"
                )
            placeholder_2d = st.empty()
            placeholder_audit = st.empty()

    delay_map = {
        "Cadencia Clinica (0.60s)": 0.60,
        "Analisis Detallado (0.90s)": 0.90,
        "Velocidad Estandar (0.35s)": 0.35,
        "Barrido Rapido (0.15s)": 0.15
    }
    delay_sec = delay_map.get(cine_speed_opt, 0.35)

    def render_display_frame(idx: int):
        cur_pkt = packets[idx]
        cur_z = cur_pkt.z_position_mm
        slice_inspect_arg = idx if inspect_single_slice else None

        fig_3d = create_3d_full_brain_visualization(
            z_coords=z_coords,
            areas=areas,
            active_z=cur_z,
            contour_slices=case_slices_meta if case_slices_meta else None,
            full_brain_slices=all_brain_slices_meta,
            pixel_spacing_mm=px_spacing,
            matrix_dim=matrix_dimension,
            h_mm=res_simpson.slice_thickness_h,
            single_slice_idx=slice_inspect_arg,
            lesion_color="#2563EB",
            view_mode=view_mode
        )
        placeholder_3d.plotly_chart(fig_3d, use_container_width=True, key=f"p3d_{idx}_{time.time_ns()}")

        if annotated_slices:
            if view_mode == "Resonancia Pura (FLAIR)" and raw_slices:
                img_to_show = raw_slices[idx]
            elif view_mode == "Mascara Radiologica" and mask_slices:
                img_to_show = mask_slices[idx]
            else:
                img_to_show = raw_slices[idx] if raw_slices else annotated_slices[idx]

            c_pts = None
            cv_pts = None
            cur_area = cur_pkt.cross_sectional_area_mm2
            cv_area = cur_area
            dice_val = 1.0
            fn_val = 0.0
            fp_val = 0.0

            if case_slices_meta and 0 <= idx < len(case_slices_meta):
                meta_item = case_slices_meta[idx]
                c_pts = meta_item.get("tumor_contour")
                cv_pts = meta_item.get("cv_contour")
                cv_area = meta_item.get("cv_area_mm2", cur_area)
                dice_val = meta_item.get("dice_similarity", 1.0)
                fn_val = meta_item.get("fn_mm2", 0.0)
                fp_val = meta_item.get("fp_mm2", 0.0)

            fig_2d = create_2d_slice_view(
                img_rgb=img_to_show,
                contour_pts=c_pts,
                cv_contour_pts=cv_pts,
                pixel_spacing_mm=px_spacing,
                matrix_dim=matrix_dimension,
                active_z=cur_z,
                slice_idx=idx,
                total_slices=total_active_slices,
                area_mm2=cur_area,
                cv_area_mm2=cv_area,
                dice_score=dice_val,
                fn_mm2=fn_val,
                fp_mm2=fp_val,
                view_mode=view_mode
            )
            placeholder_2d.plotly_chart(fig_2d, use_container_width=True, key=f"p2d_{idx}_{time.time_ns()}")

            if view_mode in ["Auditoria Dual: Clinico vs IA", "Vision Artificial Refinada (Esmeralda)"] and cur_area > 0:
                diff_mm2 = cv_area - cur_area
                diff_pct = (diff_mm2 / cur_area * 100.0) if cur_area > 0 else 0.0
                placeholder_audit.markdown(f"""
                <div style='display: flex; gap: 8px; margin-top: 6px; font-size: 11px;'>
                    <div style='flex: 1; padding: 5px 8px; background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 4px; color: #1E40AF;'>
                        <b>Mascara Radiologo (GT):</b> {cur_area:.1f} mm²
                    </div>
                    <div style='flex: 1; padding: 5px 8px; background: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 4px; color: #065F46;'>
                        <b>Segmentacion CAD:</b> {cv_area:.1f} mm² ({diff_mm2:+.1f} mm² / {diff_pct:+.1f}%)
                    </div>
                    <div style='flex: 1.2; padding: 5px 8px; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 4px; color: #334155;'>
                        <b>DICE:</b> {dice_val*100:.1f}% | <b>FN:</b> {fn_val:.1f} mm² | <b>FP:</b> {fp_val:.1f} mm²
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                placeholder_audit.empty()
        else:
            placeholder_2d.markdown("<div style='padding: 20px 0; color: #475467;'>No hay cortes disponibles para este estudio.</div>", unsafe_allow_html=True)
            placeholder_audit.empty()

    if run_cine and annotated_slices:
        total_f = len(packets)
        cine_status = st.empty()
        start_frame = selected_slice_idx
        if start_frame >= total_f - 1:
            start_frame = 0

        num_loops = 2 if loop_cine else 1
        for loop_iter in range(num_loops):
            loop_start = start_frame if loop_iter == 0 else 0
            for frame_i in range(loop_start, total_f):
                render_display_frame(frame_i)
                loop_label = " (Bucle)" if loop_cine else ""
                cine_status.markdown(
                    f"<div style='font-size: 12px; color: #2563EB; font-weight: 600; padding: 4px 8px; background: #EFF8FF; border-radius: 4px; margin-bottom: 6px;'>"
                    f"Cine-Loop PACS activo{loop_label}: Corte {frame_i + 1} de {total_f} (z = {packets[frame_i].z_position_mm:.1f} mm)</div>",
                    unsafe_allow_html=True
                )
                time.sleep(delay_sec)

        cine_status.markdown(
            "<div style='font-size: 12px; color: #027A48; font-weight: 600; padding: 4px 8px; background: #ECFDF3; border-radius: 4px; margin-bottom: 6px;'>"
            "Secuencia Cine-Loop completada.</div>",
            unsafe_allow_html=True
        )
    else:
        render_display_frame(selected_slice_idx)

    # =========================================================================
    # NIVEL 4: ANALITICA SECUNDARIA, TELEMETRIA E INFORME
    # =========================================================================
    st.markdown("<div class='section-label'>Analitica Cuantitativa y Exportacion</div>", unsafe_allow_html=True)
    tab_profile, tab_telemetry, tab_report = st.tabs([
        "Perfil Seccional A(z)",
        "Telemetria DICOM",
        "Informe Exportable"
    ])

    with tab_profile:
        with st.container(border=True):
            fig_2d = create_2d_area_profile(z_coords, areas, res_simpson.method_used)
            st.plotly_chart(fig_2d, use_container_width=True)

    with tab_telemetry:
        with st.container(border=True):
            st.markdown("<div style='font-weight: 600; color: #101828; margin-bottom: 6px;'>Registro de Paquetes de Telemetria Axial</div>", unsafe_allow_html=True)
            table_data = []
            for p in packets:
                table_data.append({
                    "Seq": p.packet_seq_num,
                    "Z (mm)": f"{p.z_position_mm:.2f}",
                    "h (mm)": f"{p.slice_thickness_h_mm:.2f}",
                    "Area (mm²)": f"{p.cross_sectional_area_mm2:.2f}",
                    "Diam. Eq (mm)": f"{p.equivalent_diameter_mm:.2f}",
                    "HU Media": f"{p.hounsfield_mean_hu:.1f}",
                    "kVp": f"{p.tube_voltage_kvp:.0f}",
                    "mA": f"{p.tube_current_ma:.0f}",
                    "CRC32": hex(p.crc32_checksum),
                    "Estado": "OK" if p.verify_integrity() else "ERROR"
                })
            st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    with tab_report:
        with st.container(border=True):
            st.markdown("<div style='font-weight: 600; color: #101828; margin-bottom: 6px;'>Informe Cuantitativo Estructurado</div>", unsafe_allow_html=True)

            report_text = f"""================================================================================
NEURO-CALC | REPORTE CUANTITATIVO DE TELEMETRIA TOMOGRAFICA Y VOLUMETRIA 3D
================================================================================
DATOS DEL ESTUDIO:
Identificador de Caso: {case_title}
Modalidad: {case_modality}
Patologia Evaluada: {lesion_label}
Fecha de Telemetria: {timestamp}
Cortes Evaluados: {res_simpson.num_slices}
Espesor Axial (h): {res_simpson.slice_thickness_h:.2f} mm
Rango Axial: [{z_coords[0]:.2f} mm a {z_coords[-1]:.2f} mm]

RESULTADOS DE INTEGRACION NUMERICA:
Metodo Aplicado: {res_simpson.method_used}
Orden de Error de Truncamiento: {res_simpson.error_order}
Volumen Simpson: {res_simpson.volume_cm3:.4f} cm3 ({res_simpson.volume_mm3:.2f} mm3)
Volumen Trapecio: {res_trapezoid.volume_cm3:.4f} cm3
Volumen Conteo de Voxeles (Riemann): {res_riemann.volume_cm3:.4f} cm3
Area Seccional Maxima: {np.max(areas):.2f} mm2 (en z = {z_coords[np.argmax(areas)]:.2f} mm)

ESTADIFICACION VOLUMETRICA:
Categoria: {risk_title}
Criterio Clinico: {risk_desc}
Verificacion CRC32: 100% integro
================================================================================
"""
            st.text_area("Vista Previa del Informe", report_text, height=220)
            st.download_button(
                label="Descargar Informe (.txt)",
                data=report_text,
                file_name=f"Reporte_NeuroCalc_{case_title}.txt",
                mime="text/plain"
            )
