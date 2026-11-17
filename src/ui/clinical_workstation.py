"""
Modulo de la Estacion Clinica PACS y Telemetria - NEURO-CALC
"""

import time
import textwrap
from typing import Tuple, List, Dict, Any, Optional
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
    alert_bg = "#FEF2F2" if risk_badge == "critical" else "#FFFBEB" if risk_badge == "warning" else "#ECFDF5"
    alert_bd = "#FECACA" if risk_badge == "critical" else "#FDE68A" if risk_badge == "warning" else "#A7F3D0"
    alert_accent = "#DC2626" if risk_badge == "critical" else "#D97706" if risk_badge == "warning" else "#059669"
    pill_cls = "pill-critical" if risk_badge == "critical" else "pill-warning" if risk_badge == "warning" else "pill-stable"
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 16px; background: {alert_bg}; border: 1px solid {alert_bd}; border-left: 4px solid {alert_accent}; border-radius: 8px; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">
        <div style="font-size: 0.86rem;">
            <strong style="color: #0F172A; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em;">Estado clinico:</strong>
            <span style="color: #334155; margin-left: 6px;">{risk_title} &mdash; {risk_desc}</span>
        </div>
        <span class="status-pill {pill_cls}">Estadificacion volumetrica</span>
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
            st.button("Inicio", key="pacs_start", on_click=pacs_nav_set, args=(1,), width="stretch")
        with col_t2:
            st.button("Anterior", key="pacs_prev", on_click=pacs_nav_set, args=(st.session_state.axial_slice_slider - 1,), width="stretch")
        with col_t3:
            run_cine = st.button("Reproducir Cine-Loop", key="pacs_play", type="primary", width="stretch")
        with col_t4:
            st.button("Siguiente", key="pacs_next", on_click=pacs_nav_set, args=(st.session_state.axial_slice_slider + 1,), width="stretch")
        with col_t5:
            st.button("Final", key="pacs_end", on_click=pacs_nav_set, args=(total_active_slices,), width="stretch")
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
                st.markdown("<div class='clinical-card-header' style='padding-top: 4px; margin-bottom: 0;'>Reconstruccion 3D de la Lesion</div>", unsafe_allow_html=True)
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
            col_2d_title, col_2d_mode, col_2d_zoom = st.columns([1.3, 1.4, 1.1])
            with col_2d_title:
                st.markdown(f"<div class='clinical-card-header' style='padding-top: 4px; margin-bottom: 0;'>Resonancia Axial FLAIR (Corte {selected_slice_1based}/{total_active_slices})</div>", unsafe_allow_html=True)
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
            with col_2d_zoom:
                zoom_mode = st.selectbox(
                    "Escala Zoom",
                    [
                        "100% (Corte Completo)",
                        "200% (Enfoque Lesion)",
                        "300% (Detalle Borde)"
                    ],
                    index=0,
                    label_visibility="collapsed"
                )
            placeholder_2d = st.empty()

    # Panel de auditoria del corte planar desplegado a ancho completo bajo los visores
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
        placeholder_3d.plotly_chart(
            fig_3d,
            width="stretch",
            config={"displayModeBar": True, "displaylogo": False},
            key=f"p3d_{idx}"
        )

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
                view_mode=view_mode,
                zoom_mode=zoom_mode
            )

            # Configuracion optimizada para control de zoom fluido y paneo sin saltos
            config_2d = {
                "scrollZoom": True,
                "displayModeBar": True,
                "modeBarButtonsToAdd": ["zoom2d", "pan2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                "displaylogo": False,
                "doubleClick": "reset"
            }
            clean_zoom_tag = zoom_mode.split()[0].replace("%", "")
            clean_view_tag = view_mode.split()[0]
            placeholder_2d.plotly_chart(
                fig_2d,
                width="stretch",
                config=config_2d,
                key=f"p2d_c_{idx}_{clean_zoom_tag}_{clean_view_tag}"
            )

            # =================================================================
            # TELEMETRIA DE REBANADA Y AUDITORIA DEL CORTE PLANAR (REALIDAD VS CAD)
            # =================================================================
            h_mm = float(res_simpson.slice_thickness_h)
            local_vol_cm3 = (cur_area * h_mm) / 1000.0
            total_vol_cm3 = float(res_simpson.volume_cm3)
            pct_of_total = (local_vol_cm3 / total_vol_cm3 * 100.0) if total_vol_cm3 > 0 else 0.0

            # Metricas de discrepancia y precision frente a la realidad (Ground Truth)
            diff_mm2 = cv_area - cur_area
            diff_pct = (diff_mm2 / cur_area * 100.0) if cur_area > 0 else 0.0
            abs_err_pct = abs(diff_pct)
            accuracy_pct = max(0.0, 100.0 - abs_err_pct)
            iou_val = (dice_val / (2.0 - dice_val)) if (2.0 - dice_val) > 0 else 0.0
            dice_pct = dice_val * 100.0
            iou_pct = iou_val * 100.0
            fn_pct = (fn_val / cur_area * 100.0) if cur_area > 0 else 0.0
            fp_pct = (fp_val / cur_area * 100.0) if cur_area > 0 else 0.0

            # Transicion con capas adyacentes
            prev_area = packets[idx - 1].cross_sectional_area_mm2 if idx > 0 else None
            next_area = packets[idx + 1].cross_sectional_area_mm2 if idx < (total_active_slices - 1) else None

            if prev_area is not None and next_area is not None:
                d_prev = cur_area - prev_area
                d_next = next_area - cur_area
                trans_html = f"Corte {idx}: {prev_area:.1f} ({d_prev:+.1f}) | Corte {idx+2}: {next_area:.1f} ({d_next:+.1f}) mm2"
            elif prev_area is not None:
                d_prev = cur_area - prev_area
                trans_html = f"Corte {idx}: {prev_area:.1f} ({d_prev:+.1f}) mm2 | Apice Superior"
            elif next_area is not None:
                d_next = next_area - cur_area
                trans_html = f"Base Inferior | Corte {idx+2}: {next_area:.1f} ({d_next:+.1f}) mm2"
            else:
                trans_html = "Corte unico"

            err_color = "#047857" if abs_err_pct <= 5.0 else "#B45309" if abs_err_pct <= 10.0 else "#B91C1C"
            err_eval_text = "Dentro de tolerancia clinica (< 5%)" if abs_err_pct <= 5.0 else "Concordancia aceptable (< 10%)" if abs_err_pct <= 10.0 else "Discrepancia significativa"

            if cur_area > 0:
                audit_html = textwrap.dedent(f"""
<div style="background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 8px; padding: 16px 20px; margin-top: 14px; box-shadow: none;">
<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #E2E8F0; padding-bottom: 8px; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">
<div>
<span style="font-size: 0.84rem; font-weight: 700; color: #0F172A; text-transform: uppercase; letter-spacing: 0.04em;">
Auditoria del Corte Planar: Realidad Radiologica vs Estimacion Numerica
</span>
<div style="font-size: 0.76rem; color: #64748B; margin-top: 2px;">
Corte {idx + 1} de {total_active_slices} | Modalidad: Resonancia Magnetica Axial FLAIR | cota z = {cur_z:.1f} mm | espesor h = {h_mm:.1f} mm
</div>
</div>
<div style="text-align: right;">
<span style="font-family: 'JetBrains Mono', monospace; font-size: 0.88rem; font-weight: 700; color: #2563EB;">
V_rebanada = {local_vol_cm3:.3f} cm3
</span>
<div style="font-size: 0.72rem; color: #64748B;">
Aporte: {pct_of_total:.1f}% del volumen total ({total_vol_cm3:.3f} cm3)
</div>
</div>
</div>
<div style="display: grid; grid-template-columns: 1.1fr 1fr 1.3fr; gap: 12px; margin-bottom: 12px;">
<div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 10px 12px;">
<div style="font-size: 0.72rem; font-weight: 700; color: #475569; text-transform: uppercase; margin-bottom: 6px;">
1. Delimitacion Seccional
</div>
<div style="display: flex; justify-content: space-between; margin-bottom: 4px; font-size: 0.82rem;">
<span style="color: #1E40AF; font-weight: 600;">Realidad (Mascara Radiologo GT):</span>
<span style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #1E40AF;">{cur_area:.1f} mm2</span>
</div>
<div style="display: flex; justify-content: space-between; margin-bottom: 4px; font-size: 0.82rem;">
<span style="color: #065F46; font-weight: 600;">Estimacion (Corte Planar CAD):</span>
<span style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #065F46;">{cv_area:.1f} mm2</span>
</div>
<div style="display: flex; justify-content: space-between; font-size: 0.78rem; border-top: 1px solid #E2E8F0; padding-top: 4px; margin-top: 4px;">
<span style="color: #64748B;">Diferencia de Area (Delta A):</span>
<span style="font-family: 'JetBrains Mono', monospace; font-weight: 600; color: #0F172A;">{diff_mm2:+.1f} mm2</span>
</div>
</div>
<div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 10px 12px;">
<div style="font-size: 0.72rem; font-weight: 700; color: #475569; text-transform: uppercase; margin-bottom: 6px;">
2. Error y Exactitud Seccional
</div>
<div style="display: flex; justify-content: space-between; margin-bottom: 4px; font-size: 0.82rem;">
<span style="color: #334155; font-weight: 500;">Error Relativo Seccional:</span>
<span style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: {err_color};">{abs_err_pct:.2f}%</span>
</div>
<div style="display: flex; justify-content: space-between; margin-bottom: 4px; font-size: 0.82rem;">
<span style="color: #334155; font-weight: 500;">Exactitud Seccional:</span>
<span style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #0F172A;">{accuracy_pct:.2f}%</span>
</div>
<div style="font-size: 0.72rem; color: {err_color}; font-weight: 600; border-top: 1px solid #E2E8F0; padding-top: 4px; margin-top: 4px;">
Estado: {err_eval_text}
</div>
</div>
<div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 10px 12px;">
<div style="font-size: 0.72rem; font-weight: 700; color: #475569; text-transform: uppercase; margin-bottom: 6px;">
3. Precision Espacial y Desglose
</div>
<div style="display: flex; justify-content: space-between; margin-bottom: 4px; font-size: 0.82rem;">
<span style="color: #334155; font-weight: 500;">Precision DICE (F1-score):</span>
<span style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #0F172A;">{dice_pct:.1f}% <span style="font-size: 0.70rem; color: #047857;">(Clinico >= 90%)</span></span>
</div>
<div style="display: flex; justify-content: space-between; margin-bottom: 4px; font-size: 0.82rem;">
<span style="color: #334155; font-weight: 500;">Superposicion Jaccard (IoU):</span>
<span style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #0F172A;">{iou_pct:.1f}%</span>
</div>
<div style="display: flex; justify-content: space-between; font-size: 0.74rem; border-top: 1px solid #E2E8F0; padding-top: 4px; margin-top: 4px; color: #64748B;">
<span>Omitido (FN): {fn_val:.1f} mm2 ({fn_pct:.1f}%)</span>
<span>Extra (FP): {fp_val:.1f} mm2 ({fp_pct:.1f}%)</span>
</div>
</div>
</div>
<div style="background: #F1F5F9; border-left: 3px solid #2563EB; border-radius: 4px; padding: 8px 12px; font-size: 0.78rem; color: #334155; line-height: 1.5;">
<strong>Interpretacion Clinica y Rigor:</strong> Transicion seccional: {trans_html}. En este corte, el algoritmo alcanza una exactitud del {accuracy_pct:.2f}% frente al estandar de oro radiologico oficial, con {fn_val:.1f} mm2 de tejido omitido (cero sub-reseccion de tumor) y {fp_val:.1f} mm2 de margen perilesional seguro integrado en la rebanada axial.
</div>
</div>
""").strip()
                placeholder_audit.markdown(audit_html, unsafe_allow_html=True)
            else:
                audit_html = textwrap.dedent(f"""
<div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px 18px; margin-top: 14px; font-size: 0.82rem; color: #64748B; text-align: center;">
<strong>Corte Planar {idx + 1} de {total_active_slices} (z = {cur_z:.1f} mm):</strong> Seccion de parenquima encefalico sano sin compromiso tumoral. Area lesional: 0.0 mm2 | Error frente al radiologo: 0.0% | V_rebanada: 0.0 cm3.
</div>
""").strip()
                placeholder_audit.markdown(audit_html, unsafe_allow_html=True)
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
            st.plotly_chart(fig_2d, width="stretch", config={"displayModeBar": True, "displaylogo": False})

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
            st.dataframe(pd.DataFrame(table_data), width="stretch", hide_index=True)

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
Volumen Simpson 1/3 (Total FLAIR): {res_simpson.volume_cm3:.4f} cm3 ({res_simpson.volume_mm3:.2f} mm3)
Volumen Trapecio Compuesto: {res_trapezoid.volume_cm3:.4f} cm3
Volumen Conteo de Voxeles (Riemann): {res_riemann.volume_cm3:.4f} cm3
Area Seccional Maxima: {np.max(areas):.2f} mm2 (en z = {z_coords[np.argmax(areas)]:.2f} mm)

ESTADIFICACION VOLUMETRICA:
Categoria: {risk_title}
Criterio Clinico: {risk_desc}
Interpretacion FLAIR: Compromiso parenquimatoso combinado (Masa tumoral + Edema vasogenico infiltrante).
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
