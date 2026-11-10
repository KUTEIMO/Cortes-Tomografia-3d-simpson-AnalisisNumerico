"""
Aplicacion Principal y Enrutador Modular - NEURO-CALC
Estacion de Telemetria Radiologica, Segmentacion y Volumetria Tomografica 3D.
"""

import sys
import os
import time
import hashlib
import hmac
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import json
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
import cv2

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.database.db_manager import PatientDatabaseManager
from src.engine.simpson_engine import NumericalEngine
from src.engine.error_evaluator import ErrorEvaluator
from src.engine.romberg_engine import RombergEngine
from src.telemetry.telemetry_packet import TelemetryPacket
from src.telemetry.stream_simulator import TelemetryStreamSimulator
from src.vision.segmenter import BrainLesionSegmenter

from src.ui.styles import CLINICAL_CSS
from src.ui.components import render_metric_card, render_workstation_header
from src.ui.clinical_workstation import render_clinical_workstation
from src.ui.academic_dossier import render_academic_dossier
from src.ui.patients_directory import render_patients_directory
from src.ui.patient_intake import render_patient_intake
from src.ui.roles_management import render_roles_and_audit

# 1. Configuracion de la ventana de Streamlit
st.set_page_config(
    page_title="NEURO-CALC | Sistema de Telemetria Tomografica",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Inyeccion del sistema de diseno claro Braintrust
st.markdown(CLINICAL_CSS, unsafe_allow_html=True)

# =============================================================================
# CONTROL DE ACCESO PERIMETRAL (SHA-256 ZERO-TRUST GATEWAY)
# =============================================================================
VALID_KEY_HASHES = {
    "13dfa397b2c1888ff0b0b6dc1d4d34fd9279c9484f59b27964e1726d91198a56",
    "78ffda519c7d0c16cc76ae31206ca8afd41f133cd20c80a7b7007ee81128eafa"
}


def _verify_access_key(raw_key: str) -> bool:
    candidate = raw_key.strip()
    if not candidate:
        return False
    # Verificar clave personalizada en st.secrets si existe en Streamlit Cloud
    try:
        if "ACCESS_KEY" in st.secrets:
            if hmac.compare_digest(candidate, str(st.secrets["ACCESS_KEY"])):
                return True
    except Exception:
        pass
    digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
    return any(hmac.compare_digest(digest, h) for h in VALID_KEY_HASHES)


if "is_presentation_authenticated" not in st.session_state:
    st.session_state.is_presentation_authenticated = False
if "auth_failed_attempts" not in st.session_state:
    st.session_state.auth_failed_attempts = 0
if "auth_lockout_until" not in st.session_state:
    st.session_state.auth_lockout_until = 0.0

if not st.session_state.is_presentation_authenticated:
    # Ocultar barra lateral por completo en la vista de inicio de sesion
    st.markdown("""
    <style>
        section[data-testid="stSidebar"] { display: none !important; }
        [data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

    now_ts = time.time()
    if now_ts < st.session_state.auth_lockout_until:
        remaining_lock = int(st.session_state.auth_lockout_until - now_ts)
        st.error(f"Acceso bloqueado temporalmente. Reintente en {remaining_lock} segundos.")
        st.stop()

    col_auth_center = st.columns([1.2, 1.0, 1.2])[1]
    with col_auth_center:
        st.markdown("""
        <div style="margin-top: 64px; background: #FFFFFF; border: 1px solid #EAECF0; border-radius: 10px; padding: 28px 24px 20px 24px; text-align: center;">
            <div style="font-size: 0.68rem; font-weight: 700; color: #2563EB; letter-spacing: 0.08em; text-transform: uppercase;">
                ACCESO RESTRINGIDO
            </div>
            <h2 style="margin: 6px 0 4px 0; font-size: 1.35rem; color: #101828;">NEURO-CALC</h2>
            <div style="font-size: 0.80rem; color: #667085;">
                Estacion de Telemetria Tomografica
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("auth_login_form", clear_on_submit=True):
            entered_key = st.text_input(
                "Clave de Acceso",
                type="password",
                placeholder="Clave de acceso institucional",
                label_visibility="collapsed"
            )
            submitted = st.form_submit_button("Ingresar", type="primary", width="stretch")

            if submitted:
                if _verify_access_key(entered_key):
                    st.session_state.is_presentation_authenticated = True
                    st.session_state.auth_failed_attempts = 0
                    st.rerun()
                else:
                    st.session_state.auth_failed_attempts += 1
                    if st.session_state.auth_failed_attempts >= 5:
                        st.session_state.auth_lockout_until = time.time() + 60.0
                        st.error("Demasiados intentos fallidos. Bloqueado por 60 segundos.")
                    else:
                        rem = 5 - st.session_state.auth_failed_attempts
                        st.error(f"Clave invalida. Intentos restantes: {rem}.")
    st.stop()

# 3. Inicializacion del Gestor de Base de Datos y Sesion
db = PatientDatabaseManager()

if "current_user_id" not in st.session_state:
    st.session_state.current_user_id = "MED_PRINCIPAL"

if "active_nav_route" not in st.session_state:
    st.session_state.active_nav_route = "Directorio de Pacientes"

all_patients = db.list_patients()
patient_ids = [p["patient_id"] for p in all_patients]

# Aplicar cambio pendiente de paciente ANTES de que ningun widget sea instanciado
if "pending_patient_switch" in st.session_state:
    target_pid = st.session_state.pop("pending_patient_switch")
    st.session_state.active_patient_id = target_pid
    st.session_state.side_pacs_patient_switcher = target_pid
    st.session_state.top_main_pacs_switcher = target_pid
    if "axial_slice_slider" in st.session_state:
        del st.session_state["axial_slice_slider"]

if "active_patient_id" not in st.session_state or st.session_state.active_patient_id not in patient_ids:
    st.session_state.active_patient_id = patient_ids[0] if patient_ids else "PATIENT_01_TCGA_4944"

# Asegurar sincronizacion inicial antes de renderizar widgets
if "side_pacs_patient_switcher" not in st.session_state or st.session_state.side_pacs_patient_switcher not in patient_ids:
    st.session_state.side_pacs_patient_switcher = st.session_state.active_patient_id

if "top_main_pacs_switcher" not in st.session_state or st.session_state.top_main_pacs_switcher not in patient_ids:
    st.session_state.top_main_pacs_switcher = st.session_state.active_patient_id


def on_side_patient_change():
    new_pid = st.session_state.side_pacs_patient_switcher
    st.session_state.active_patient_id = new_pid
    st.session_state.top_main_pacs_switcher = new_pid
    if "axial_slice_slider" in st.session_state:
        del st.session_state["axial_slice_slider"]


def on_top_patient_change():
    new_pid = st.session_state.top_main_pacs_switcher
    st.session_state.active_patient_id = new_pid
    st.session_state.side_pacs_patient_switcher = new_pid
    if "axial_slice_slider" in st.session_state:
        del st.session_state["axial_slice_slider"]


def set_active_patient(patient_id: str):
    st.session_state.active_patient_id = patient_id
    st.session_state.pending_patient_switch = patient_id
    if "axial_slice_slider" in st.session_state:
        del st.session_state["axial_slice_slider"]
    st.session_state.active_nav_route = "Estacion PACS & Resultados"
    st.rerun()


def set_active_route(route_name: str):
    st.session_state.active_nav_route = route_name
    st.rerun()


def set_active_user(user_id: str):
    st.session_state.current_user_id = user_id
    st.rerun()


p_options = {p["patient_id"]: f"{p['patient_code']} - {p['pathology'][:26]}" for p in all_patients}
active_user_data = db.get_user(st.session_state.current_user_id) or db.list_users()[0]

# =============================================================================
# 4. ENCABEZADO INSTITUCIONAL Y NAVEGACION PRINCIPAL (4 RUTAS)
# =============================================================================
current_route = st.session_state.active_nav_route
if current_route in ("Dossier Cientifico & Teoria", "Roles y Auditoria Hospitalaria"):
    current_route = "Administracion"
    st.session_state.active_nav_route = "Administracion"

with st.container(border=True):
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; padding-bottom: 10px; border-bottom: 1px solid #F1F5F9; margin-bottom: 10px;">
        <div style="display: flex; align-items: baseline; gap: 10px;">
            <span style="font-size: 1.18rem; font-weight: 700; color: #0F172A; letter-spacing: -0.02em;">NEURO-CALC</span>
            <span style="font-size: 0.82rem; color: #64748B;">Estacion de volumetria tomografica</span>
        </div>
        <div style="display: flex; align-items: center; gap: 12px; font-size: 0.82rem; color: #475469;">
            <span>{active_user_data['full_name']} &middot; <span style="color: #64748B;">{active_user_data['role']}</span></span>
            <span class="status-pill pill-stable">Sesion activa</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_n1, col_n2, col_n3, col_n4 = st.columns(4)
    nav_tabs = [
        ("Directorio de Pacientes", "Directorio", col_n1),
        ("Nuevo Paciente e Ingesta", "Nuevo paciente", col_n2),
        ("Estacion PACS & Resultados", "Estacion PACS", col_n3),
        ("Administracion", "Administracion", col_n4)
    ]

    for route_key, label_text, col_target in nav_tabs:
        with col_target:
            is_sel = (current_route == route_key)
            b_type = "primary" if is_sel else "secondary"
            if st.button(label_text, key=f"top_btn_{route_key}", type=b_type, width="stretch"):
                st.session_state.active_nav_route = route_key
                st.rerun()

# 5. Barra Lateral (Navegacion Rapida y Conmutador)
with st.sidebar:
    st.markdown("""
    <div style="padding-bottom: 12px; border-bottom: 1px solid #E2E8F0; margin-bottom: 14px;">
        <div style="font-size: 0.68rem; font-weight: 700; color: #64748B; letter-spacing: 0.06em; text-transform: uppercase;">
            Estacion Diagnostica
        </div>
        <h3 style="margin: 2px 0 0 0; font-size: 1.15rem; color: #0F172A;">NEURO-CALC</h3>
        <div style="font-size: 0.76rem; color: #64748B; margin-top: 2px;">
            Volumetria por Cuadratura O(h⁴)
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size: 0.72rem; font-weight: 700; color: #475469; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px;'>Navegacion</div>", unsafe_allow_html=True)
    for route_key, label_text, _ in nav_tabs:
        is_sel = (current_route == route_key)
        if st.button(label_text, key=f"side_btn_{route_key}", type="primary" if is_sel else "secondary", width="stretch"):
            st.session_state.active_nav_route = route_key
            st.rerun()

    st.markdown("<hr style='border-color: #E2E8F0; margin: 14px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.72rem; font-weight: 700; color: #475469; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px;'>Paciente activo</div>", unsafe_allow_html=True)
    st.selectbox(
        "Paciente activo",
        options=list(p_options.keys()),
        format_func=lambda x: p_options[x],
        key="side_pacs_patient_switcher",
        on_change=on_side_patient_change,
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color: #E2E8F0; margin: 14px 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 6px; padding: 8px 10px; margin-bottom: 10px;">
        <div style="font-size: 0.70rem; font-weight: 700; color: #065F46;">SESION ACTIVA</div>
        <div style="font-size: 0.72rem; color: #047857;">Acceso autenticado.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Cerrar sesion", key="side_lock_btn", width="stretch"):
        st.session_state.is_presentation_authenticated = False
        st.rerun()


# =============================================================================
# 6. ENRUTAMIENTO Y DESPLIEGUE MODULAR DE VISTAS
# =============================================================================

# -----------------------------------------------------------------------------
# RUTA 1: DIRECTORIO CLINICO DE PACIENTES
# -----------------------------------------------------------------------------
if current_route == "Directorio de Pacientes":
    render_patients_directory(
        db=db,
        current_user_id=st.session_state.current_user_id,
        on_select_patient=set_active_patient,
        on_navigate_intake=lambda: set_active_route("Nuevo Paciente e Ingesta")
    )

# -----------------------------------------------------------------------------
# RUTA 2: NUEVO PACIENTE E INGESTA RADIOLOGICA
# -----------------------------------------------------------------------------
elif current_route == "Nuevo Paciente e Ingesta":
    render_patient_intake(
        db=db,
        current_user_id=st.session_state.current_user_id,
        on_success_intake=set_active_patient
    )

# -----------------------------------------------------------------------------
# RUTA 3: ESTACION PACS & RESULTADOS
# -----------------------------------------------------------------------------
elif current_route == "Estacion PACS & Resultados":
    active_pid = st.session_state.active_patient_id
    patient_record = db.get_patient(active_pid)

    if not patient_record:
        st.warning("Seleccione un paciente valido en el Directorio para desplegar la Estacion PACS.")
        if st.button("Ir al Directorio de Pacientes"):
            set_active_route("Directorio de Pacientes")
    else:
        col_pacs_h1, col_pacs_h2 = st.columns([3.2, 1.8])
        with col_pacs_h1:
            st.markdown(f"""
            <div class="page-header-wrap" style="margin-bottom: 10px;">
                <div class="page-eyebrow">Estacion de Telemetria PACS</div>
                <h1 class="page-title" style="font-size: 1.55rem !important;">{patient_record.get('patient_code', '')} &middot; {patient_record.get('pathology', '')}</h1>
                <p class="page-subtitle">{patient_record.get('full_name', '')} &middot; {patient_record.get('modality', '')}</p>
            </div>
            """, unsafe_allow_html=True)
        with col_pacs_h2:
            st.markdown("<div style='font-size: 0.74rem; font-weight: 600; color: #64748B; margin-top: 14px; margin-bottom: 4px;'>CAMBIAR ESTUDIO ACTIVO</div>", unsafe_allow_html=True)
            st.selectbox(
                "Conmutar Paciente Directo",
                options=list(p_options.keys()),
                format_func=lambda x: p_options[x],
                key="top_main_pacs_switcher",
                on_change=on_top_patient_change,
                label_visibility="collapsed"
            )

        # Carga de datos de la serie segun el caso seleccionado
        packets = []
        annotated_slices = []
        raw_slices = []
        mask_slices = []
        case_slices_meta = []
        all_brain_slices_meta = None
        px_spacing = patient_record.get("pixel_spacing", 0.9375)
        matrix_dimension = patient_record.get("matrix_dim", 256)
        h_val = patient_record.get("slice_thickness_h", 5.0)

        case_dir_rel = patient_record.get("data_dir", "")
        case_dir_abs = BASE_DIR / case_dir_rel if case_dir_rel != "synthetic" else None

        if case_dir_abs and case_dir_abs.exists():
            # Procesar carpeta del caso real
            all_pkts, all_annot, info = BrainLesionSegmenter.process_case_series(
                str(case_dir_abs),
                lesion_type="Glioma" if "Astrocitoma" in patient_record.get("pathology", "") else "Tumor"
            )
            px_spacing = info.get("pixel_spacing_mm", px_spacing)
            matrix_dimension = info.get("matrix_dim", matrix_dimension)
            all_brain_slices_meta = info.get("slices", [])

            # Ventana tumoral delimitada en TCGA segun la serie activa
            if "TCGA_4944" in active_pid and len(all_pkts) >= 15:
                s_start, s_end = 4, 15
                packets = all_pkts[s_start:s_end]
                annotated_slices = all_annot[s_start:s_end]
                raw_slices = info["raw_images"][s_start:s_end]
                mask_slices = info["mask_images"][s_start:s_end]
                case_slices_meta = info["slices"][s_start:s_end]
            elif "TCGA_4941" in active_pid and len(all_pkts) >= 19:
                s_start, s_end = 9, 19
                packets = all_pkts[s_start:s_end]
                annotated_slices = all_annot[s_start:s_end]
                raw_slices = info["raw_images"][s_start:s_end]
                mask_slices = info["mask_images"][s_start:s_end]
                case_slices_meta = info["slices"][s_start:s_end]
            elif "TCGA_4943" in active_pid and len(all_pkts) >= 20:
                s_start, s_end = 9, 20
                packets = all_pkts[s_start:s_end]
                annotated_slices = all_annot[s_start:s_end]
                raw_slices = info["raw_images"][s_start:s_end]
                mask_slices = info["mask_images"][s_start:s_end]
                case_slices_meta = info["slices"][s_start:s_end]
            elif "TCGA_5393" in active_pid and len(all_pkts) >= 14:
                s_start, s_end = 3, 14
                packets = all_pkts[s_start:s_end]
                annotated_slices = all_annot[s_start:s_end]
                raw_slices = info["raw_images"][s_start:s_end]
                mask_slices = info["mask_images"][s_start:s_end]
                case_slices_meta = info["slices"][s_start:s_end]
            elif "TCGA_4942" in active_pid and len(all_pkts) >= 15:
                s_start, s_end = 7, 15
                packets = all_pkts[s_start:s_end]
                annotated_slices = all_annot[s_start:s_end]
                raw_slices = info["raw_images"][s_start:s_end]
                mask_slices = info["mask_images"][s_start:s_end]
                case_slices_meta = info["slices"][s_start:s_end]
            elif "TCGA_7684" in active_pid and len(all_pkts) >= 23:
                s_start, s_end = 11, 23
                packets = all_pkts[s_start:s_end]
                annotated_slices = all_annot[s_start:s_end]
                raw_slices = info["raw_images"][s_start:s_end]
                mask_slices = info["mask_images"][s_start:s_end]
                case_slices_meta = info["slices"][s_start:s_end]
            else:
                packets = all_pkts
                annotated_slices = all_annot
                raw_slices = info.get("raw_images", [])
                mask_slices = info.get("mask_images", [])
                case_slices_meta = info.get("slices", [])
        else:
            # Fantoma esferico analitico con cortes 2D sintetizados
            sim = TelemetryStreamSimulator()
            packets = sim.generate_synthetic_cyst_stream(15, 2.67, 20.0, 20.0, 20.0)
            dim = 256
            px_spacing = 1.0
            matrix_dimension = dim
            h_val = 2.67
            annotated_slices = []
            raw_slices = []
            mask_slices = []
            case_slices_meta = []

            for idx, p in enumerate(packets):
                cur_z = p.z_position_mm
                r_z = np.sqrt(max(0.0, 20.0**2 - cur_z**2))
                img = np.zeros((dim, dim, 3), dtype=np.uint8)
                # Contenedor cilindrico de fantoma acrilico estandar ACR
                cv2.circle(img, (128, 128), 108, (38, 44, 52), -1)   # Solucion acuosa salina de contraste
                cv2.circle(img, (128, 128), 108, (110, 122, 138), 2) # Pared acrilica exterior
                # Reticula de calibracion milimetrica
                for g_off in [-60, -30, 30, 60]:
                    cv2.line(img, (128 + g_off, 35), (128 + g_off, 221), (50, 58, 68), 1)
                    cv2.line(img, (35, 128 + g_off), (221, 128 + g_off), (50, 58, 68), 1)
                # Esfera de prueba interna (radio R=20mm)
                if r_z > 0:
                    cv2.circle(img, (128, 128), int(round(r_z)), (25, 75, 150), -1)
                    cv2.circle(img, (128, 128), int(round(r_z)), (37, 99, 235), 2)
                mask = np.zeros((dim, dim), dtype=np.uint8)
                if r_z > 0:
                    cv2.circle(mask, (128, 128), int(round(r_z)), 255, -1)

                thetas = np.linspace(0, 2 * np.pi, 36, endpoint=False)
                c_pts = [[128.0 + r_z * np.cos(th), 128.0 + r_z * np.sin(th)] for th in thetas] if r_z > 0 else []

                annotated_slices.append(img)
                raw_slices.append(img)
                mask_slices.append(mask)
                case_slices_meta.append({
                    "slice_idx": idx,
                    "z_position_mm": cur_z,
                    "slice_thickness_h_mm": h_val,
                    "tumor_contour": c_pts,
                    "area_mm2": p.cross_sectional_area_mm2,
                    "cv_contour": c_pts,
                    "cv_area_mm2": p.cross_sectional_area_mm2,
                    "dice_similarity": 1.0
                })
            all_brain_slices_meta = case_slices_meta

        if not packets:
            st.error("No se pudieron cargar paquetes de telemetria para este paciente.")
        else:
            z_coords = np.array([p.z_position_mm for p in packets], dtype=np.float64)
            areas = np.array([p.cross_sectional_area_mm2 for p in packets], dtype=np.float64)

            res_simpson = NumericalEngine.calculate_simpson_13_composite(z_coords, areas)
            res_trapezoid = NumericalEngine.calculate_trapezoidal(z_coords, areas)
            res_riemann = NumericalEngine.calculate_voxel_counting_riemann(z_coords, areas)
            romberg_data = RombergEngine.compute_romberg_table(z_coords, areas, max_levels=3)

            # Resumen volumetrico y metodo aplicado
            timestamp = packets[0].timestamp_utc if packets else "2026-11-06T12:00:00Z"
            crc_all_ok = all(p.verify_integrity() for p in packets)
            method_desc = "Simpson 1/3 compuesto (par)" if "Puro" in res_simpson.method_used else "Hibrido Simpson 1/3 + 3/8 (impar)"

            col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
            with col_kpi1:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Volumen calculado</div>
                    <div class="kpi-value">{res_simpson.volume_cm3:.3f}<span class="kpi-unit">cm³</span></div>
                    <div class="kpi-context">{method_desc}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_kpi2:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Orden de convergencia</div>
                    <div class="kpi-value">{res_simpson.error_order}</div>
                    <div class="kpi-context">Cuadratura polinomial compuesta</div>
                </div>
                """, unsafe_allow_html=True)
            with col_kpi3:
                crc_label = "CRC32 integro al 100%" if crc_all_ok else "Error de verificacion CRC32"
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Adquisicion e integridad</div>
                    <div class="kpi-value">{res_simpson.num_slices} cortes<span class="kpi-unit">&middot; {res_simpson.slice_thickness_h:.1f} mm</span></div>
                    <div class="kpi-context">{crc_label}</div>
                </div>
                """, unsafe_allow_html=True)

            render_clinical_workstation(
                packets=packets,
                annotated_slices=annotated_slices,
                raw_slices=raw_slices,
                mask_slices=mask_slices,
                case_slices_meta=case_slices_meta,
                all_brain_slices_meta=all_brain_slices_meta,
                res_simpson=res_simpson,
                res_trapezoid=res_trapezoid,
                res_riemann=res_riemann,
                z_coords=z_coords,
                areas=areas,
                px_spacing=px_spacing,
                matrix_dimension=matrix_dimension,
                case_title=patient_record.get("patient_code", active_pid),
                case_modality=patient_record.get("modality", "MR"),
                lesion_label=patient_record.get("pathology", "Neoplasia"),
                timestamp=timestamp
            )

# -----------------------------------------------------------------------------
# RUTA 4: ADMINISTRACION (ESPECIFICACION TECNICA Y ROLES RBAC)
# -----------------------------------------------------------------------------
elif current_route == "Administracion":
    st.markdown("""
    <div class="page-header-wrap">
        <div class="page-eyebrow">Administracion del Sistema</div>
        <h1 class="page-title">Especificacion tecnica y auditoria</h1>
        <p class="page-subtitle">Documentacion matematica del motor volumetrico, cuantificacion de error y control de acceso.</p>
    </div>
    """, unsafe_allow_html=True)

    tab_adm_spec, tab_adm_rbac = st.tabs([
        "Especificacion tecnica y error",
        "Roles y auditoria (RBAC)"
    ])

    with tab_adm_spec:
        active_pid = st.session_state.active_patient_id
        patient_record = db.get_patient(active_pid) or db.list_patients()[0]
        case_dir_abs = BASE_DIR / patient_record.get("data_dir", "data/kaggle_cases/PATIENT_01_TCGA_4944")

        if case_dir_abs.exists():
            all_pkts, _, info = BrainLesionSegmenter.process_case_series(
                str(case_dir_abs),
                lesion_type="Glioma" if "Astrocitoma" in patient_record.get("pathology", "") else "Tumor"
            )
            if "TCGA_4944" in active_pid and len(all_pkts) >= 15:
                packets = all_pkts[4:15]
                case_slices_meta = info.get("slices", [])[4:15]
            elif "TCGA_4941" in active_pid and len(all_pkts) >= 19:
                packets = all_pkts[9:19]
                case_slices_meta = info.get("slices", [])[9:19]
            elif "TCGA_4943" in active_pid and len(all_pkts) >= 20:
                packets = all_pkts[9:20]
                case_slices_meta = info.get("slices", [])[9:20]
            elif "TCGA_5393" in active_pid and len(all_pkts) >= 14:
                packets = all_pkts[3:14]
                case_slices_meta = info.get("slices", [])[3:14]
            elif "TCGA_4942" in active_pid and len(all_pkts) >= 15:
                packets = all_pkts[7:15]
                case_slices_meta = info.get("slices", [])[7:15]
            elif "TCGA_7684" in active_pid and len(all_pkts) >= 23:
                packets = all_pkts[11:23]
                case_slices_meta = info.get("slices", [])[11:23]
            else:
                packets = all_pkts
                case_slices_meta = info.get("slices", [])
            z_coords = np.array([p.z_position_mm for p in packets], dtype=np.float64)
            areas = np.array([p.cross_sectional_area_mm2 for p in packets], dtype=np.float64)
        else:
            sim = TelemetryStreamSimulator()
            packets = sim.generate_synthetic_cyst_stream(15, 2.67, 20.0, 20.0, 20.0)
            z_coords = np.array([p.z_position_mm for p in packets], dtype=np.float64)
            areas = np.array([p.cross_sectional_area_mm2 for p in packets], dtype=np.float64)
            case_slices_meta = []

        res_simpson = NumericalEngine.calculate_simpson_13_composite(z_coords, areas)
        res_trapezoid = NumericalEngine.calculate_trapezoidal(z_coords, areas)
        res_riemann = NumericalEngine.calculate_voxel_counting_riemann(z_coords, areas)
        romberg_data = RombergEngine.compute_romberg_table(z_coords, areas, max_levels=3)

        render_academic_dossier(
            res_simpson=res_simpson,
            res_trapezoid=res_trapezoid,
            res_riemann=res_riemann,
            romberg_data=romberg_data,
            case_slices_meta=case_slices_meta,
            z_coords=z_coords,
            areas=areas,
            dataset_option=patient_record.get("pathology", "Astrocitoma"),
            patient_record=patient_record,
            all_patients=all_patients,
            embedded=True
        )

    with tab_adm_rbac:
        render_roles_and_audit(
            db=db,
            current_user_id=st.session_state.current_user_id,
            on_change_user=set_active_user,
            embedded=True
        )
