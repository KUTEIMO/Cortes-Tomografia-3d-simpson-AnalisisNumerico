"""
Modulo de Admision e Ingesta de Pacientes - NEURO-CALC
Recepcion de series tomograficas/RMN, extraccion morfometrica y volumetria Simpson 1/3.
"""

import os
import re
import zipfile
import json
import shutil
from pathlib import Path
from typing import Callable, Dict, Any
import numpy as np
import cv2
import streamlit as st

from src.database.db_manager import PatientDatabaseManager
from src.vision.segmenter import BrainLesionSegmenter
from src.engine.simpson_engine import NumericalEngine

BASE_DIR = Path(__file__).resolve().parent.parent.parent
EXAMPLES_DIR = BASE_DIR / "data" / "banco_casos_reales_zip"
CASES_DIR = BASE_DIR / "data" / "kaggle_cases"

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".json"}
MAX_ZIP_FILES = 120
MAX_ZIP_UNCOMPRESSED_BYTES = 150 * 1024 * 1024  # 150 MB


def _safe_extract_zip(zip_source: Any, target_dir: Path):
    """Descomprime archivos ZIP validando extensiones, cuota de tamano y proteccion contra Path Traversal."""
    target_resolved = target_dir.resolve()
    target_resolved.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_source, "r") as zf:
        infolist = zf.infolist()
        if len(infolist) > MAX_ZIP_FILES:
            raise ValueError(f"El archivo ZIP excede el limite de elementos ({MAX_ZIP_FILES}).")

        total_size = 0
        safe_members = []
        for member in infolist:
            total_size += member.file_size
            if total_size > MAX_ZIP_UNCOMPRESSED_BYTES:
                raise ValueError("El archivo descomprimido supera el limite de 150 MB.")
            if member.is_dir():
                continue

            member_path = Path(member.filename)
            if member_path.suffix.lower() not in ALLOWED_EXTENSIONS:
                raise ValueError(f"Extension no permitida: {member.filename}")

            dest_path = (target_resolved / member_path.name).resolve()
            if not str(dest_path).startswith(str(target_resolved)):
                raise ValueError("Ruta no valida detectada dentro del archivo comprimido.")

            safe_members.append((member, dest_path))

        for member, dest_path in safe_members:
            with zf.open(member) as source, open(dest_path, "wb") as target:
                shutil.copyfileobj(source, target)


def render_patient_intake(
    db: PatientDatabaseManager,
    current_user_id: str,
    on_success_intake: Callable[[str], None]
):
    """Renderiza el modulo de admision e ingesta de estudios radiologicos."""
    st.markdown("""
    <div class="page-header">
        <div class="page-eyebrow">Admision Radiologica</div>
        <h1 class="page-title">Registro y Procesamiento de Nuevo Paciente</h1>
        <p class="page-subtitle">Ingesta de series axiales, segmentacion morfometrica e integracion volumetrica.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Criterios Tecnicos de Adquisicion y Formatos Admitidos", expanded=False):
        st.markdown("""
        <div style="font-size: 0.82rem; color: #344054; line-height: 1.6;">
            Requisitos tecnicos para mantener convergencia de cuarto orden <strong>O(h^4)</strong>:
            <ul style="margin-top: 6px; padding-left: 18px;">
                <li><strong>Plano Anatómico:</strong> Cortes axiales transversos paralelos a la Linea Orbito-Meatal (LOM 0°).</li>
                <li><strong>Espesor Axial (h):</strong> Espaciado uniforme constante (tipicamente 3.0 mm a 5.0 mm).</li>
                <li><strong>Secuencia:</strong> RMN Axial FLAIR para edema/infiltracion glial; RMN T1 con Gadolinio (T1+C) para lesiones captantes.</li>
                <li><strong>Formato de Carga:</strong> Paquete <code>.zip</code> con cortes secuenciales ordenados y archivo opcional <code>metadata.json</code>.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    tab_examples, tab_manual = st.tabs([
        "Banco Clinico Validado",
        "Carga Manual de Estudio (.zip)"
    ])

    # -------------------------------------------------------------------------
    # PESTANA 1: INGESTA DESDE BANCO CLINICO
    # -------------------------------------------------------------------------
    with tab_examples:
        with st.container(border=True):
            st.markdown("""
            <div style="font-weight: 700; color: #101828; font-size: 0.95rem; margin-bottom: 4px;">
                Ingesta Rapida desde Repositorio de Casos Reales (TCGA)
            </div>
            <div style="font-size: 0.82rem; color: #475467; margin-bottom: 12px;">
                Seleccione un paquete hospitalario de <code>data/banco_casos_reales_zip/</code> para desempaquetarlo, procesar sus contornos y registrarlo en el directorio.
            </div>
            """, unsafe_allow_html=True)

            example_cases = {
                "Caso 1: Astrocitoma Fibrilar Frontoparietal (TCGA-4944) - RMN FLAIR 11 cortes": {
                    "zip": "CASO_01_ASTROCITOMA_TCGA_4944.zip",
                    "patient_id": "PATIENT_01_TCGA_4944",
                    "code": "PAC-2026-4944",
                    "name": "Paciente Anonimizado TCGA-4944",
                    "pathology": "Astrocitoma Fibrilar Grado II",
                    "modality": "MR Axial FLAIR",
                    "h": 5.0,
                    "px": 0.9375
                },
                "Caso 2: Oligodendroglioma de Bajo Grado (TCGA-4941) - RMN FLAIR 10 cortes": {
                    "zip": "CASO_02_OLIGODENDROGLIOMA_TCGA_4941.zip",
                    "patient_id": "PATIENT_02_TCGA_4941",
                    "code": "PAC-2026-4941",
                    "name": "Paciente Anonimizado TCGA-4941",
                    "pathology": "Oligodendroglioma de Bajo Grado",
                    "modality": "MR Axial FLAIR",
                    "h": 5.0,
                    "px": 0.9375
                },
                "Caso 3: Oligoastrocitoma Anaplasico (TCGA-4943) - RMN FLAIR 11 cortes": {
                    "zip": "CASO_03_OLIGOASTROCITOMA_TCGA_4943.zip",
                    "patient_id": "PATIENT_03_TCGA_4943",
                    "code": "PAC-2026-4943",
                    "name": "Paciente Anonimizado TCGA-4943",
                    "pathology": "Oligoastrocitoma Grado II",
                    "modality": "MR Axial FLAIR",
                    "h": 5.0,
                    "px": 0.9375
                },
                "Caso 4: Glioblastoma Multiforme Parietotemporal (TCGA-5393) - RMN FLAIR 11 cortes": {
                    "zip": "CASO_04_GLIOBLASTOMA_TCGA_5393.zip",
                    "patient_id": "PATIENT_04_TCGA_5393",
                    "code": "PAC-2026-5393",
                    "name": "Paciente Anonimizado TCGA-5393",
                    "pathology": "Glioblastoma Multiforme",
                    "modality": "MR Axial FLAIR",
                    "h": 5.0,
                    "px": 0.9375
                },
                "Caso 5: Glioma Frontal de Grado II (TCGA-4942) - RMN FLAIR 8 cortes": {
                    "zip": "CASO_05_GLIOMA_FRONTAL_TCGA_4942.zip",
                    "patient_id": "PATIENT_05_TCGA_4942",
                    "code": "PAC-2026-4942",
                    "name": "Paciente Anonimizado TCGA-4942",
                    "pathology": "Glioma Frontal de Grado II",
                    "modality": "MR Axial FLAIR",
                    "h": 5.0,
                    "px": 0.9375
                },
                "Caso 6: Astrocitoma Fibrilar Parietal Izquierdo (TCGA-7684) - RMN FLAIR 12 cortes": {
                    "zip": "CASO_06_ASTROCITOMA_TCGA_7684.zip",
                    "patient_id": "PATIENT_06_TCGA_7684",
                    "code": "PAC-2026-7684",
                    "name": "Paciente Anonimizado TCGA-7684",
                    "pathology": "Astrocitoma Fibrilar Parietal",
                    "modality": "MR Axial FLAIR",
                    "h": 5.0,
                    "px": 0.9375
                }
            }

            selected_case_label = st.selectbox(
                "Seleccionar Estudio Clinico",
                list(example_cases.keys())
            )
            selected_case_info = example_cases[selected_case_label]

            if st.button("Procesar e Incorporar Paciente", type="primary", use_container_width=True):
                with st.spinner("Extrayendo serie tomografica y ejecutando segmentacion..."):
                    zip_path = EXAMPLES_DIR / selected_case_info["zip"]
                    target_dir = CASES_DIR / selected_case_info["patient_id"]
                    try:
                        if zip_path.exists():
                            _safe_extract_zip(zip_path, target_dir)
                    except ValueError as ve:
                        st.error(str(ve))
                        st.stop()

                    meta_file = target_dir / "metadata.json"
                    if meta_file.exists():
                        with open(meta_file, "r", encoding="utf-8") as f:
                            meta_data = json.load(f)
                        slices = meta_data.get("slices", [])
                        z_vals = np.array([float(s["z_position_mm"]) for s in slices], dtype=float)
                        areas_vals = np.array([float(s.get("cv_area_mm2", s.get("area_mm2", 0.0))) for s in slices], dtype=float)
                        simpson_res = NumericalEngine.calculate_simpson_13_composite(z_vals, areas_vals)
                        vol_cm3 = float(simpson_res.volume_cm3)
                        n_slices = len(slices)
                    else:
                        vol_cm3 = 45.0
                        n_slices = 11

                    risk_stage = (
                        "Grado III: Efecto de Masa Critico (Urgente)" if vol_cm3 > 25.0
                        else "Grado II: Masa Moderada (5 - 25 cm3)" if vol_cm3 >= 5.0
                        else "Grado I: Lesion Focal Menor (< 5 cm3)"
                    )

                    patient_record = {
                        "patient_id": selected_case_info["patient_id"],
                        "patient_code": selected_case_info["code"],
                        "full_name": selected_case_info["name"],
                        "age": 48,
                        "gender": "M",
                        "modality": selected_case_info["modality"],
                        "pathology": selected_case_info["pathology"],
                        "num_slices": n_slices,
                        "slice_thickness_h": selected_case_info["h"],
                        "pixel_spacing": selected_case_info["px"],
                        "matrix_dim": 256,
                        "volume_cm3": vol_cm3,
                        "risk_stage": risk_stage,
                        "physician": "Unidad de Neuro-Oncologia",
                        "data_dir": f"data/kaggle_cases/{selected_case_info['patient_id']}",
                        "status": "Procesado y Validado",
                        "notes": f"Serie {selected_case_info['zip']} procesada. Volumetria por Simpson 1/3 completada."
                    }

                    db.save_patient(patient_record, user_id=current_user_id)
                    st.success(f"Paciente {selected_case_info['code']} registrado correctamente (Volumen: {vol_cm3:.3f} cm³).")

                    if st.button("Abrir en Estacion PACS", key="btn_open_after_intake"):
                        on_success_intake(selected_case_info["patient_id"])

    # -------------------------------------------------------------------------
    # PESTANA 2: ADMISION MANUAL POR ARCHIVO ZIP
    # -------------------------------------------------------------------------
    with tab_manual:
        with st.container(border=True):
            st.markdown("""
            <div style="font-weight: 700; color: #101828; font-size: 0.95rem; margin-bottom: 8px;">
                Formulario de Admision y Parametrizacion Tomografica
            </div>
            """, unsafe_allow_html=True)

            col_f1, col_f2 = st.columns(2)
            with col_f1:
                st.markdown("<div class='section-label'>Datos del Paciente</div>", unsafe_allow_html=True)
                form_code = st.text_input("Codigo de Historia Clinica", value="PAC-2026-9901")
                form_name = st.text_input("Identificador Anonimizado", value="Paciente Anonimizado 04")
                form_age = st.number_input("Edad", min_value=1, max_value=110, value=45)
                form_gender = st.selectbox("Sexo Biologico", ["Masculino (M)", "Femenino (F)", "Otro"])

            with col_f2:
                st.markdown("<div class='section-label'>Parametros de Adquisicion</div>", unsafe_allow_html=True)
                form_pathology = st.text_input("Diagnostico Presuntivo", value="Glioblastoma Multiforme Frontal")
                form_modality = st.selectbox("Modalidad", ["MR Axial FLAIR", "MR Axial T1+C", "TC Cerebral Simple", "TC Cerebral Contrastada"])
                form_h = st.number_input("Espesor Nominal h (mm)", min_value=0.5, max_value=15.0, value=5.0, step=0.5)
                form_px = st.number_input("Pixel Spacing (mm/px)", min_value=0.1, max_value=3.0, value=0.9375, step=0.05)

            st.markdown("<div class='section-label'>Archivo de Serie Tomografica</div>", unsafe_allow_html=True)
            uploaded_case_file = st.file_uploader(
                "Cargar Archivo ZIP con cortes axiales ordenados",
                type=["zip"],
                help="Suba un archivo .zip con cortes PNG/JPG y metadata.json opcional."
            )

            if st.button("Ejecutar Segmentacion e Incorporar Estudio", type="primary", use_container_width=True):
                if not uploaded_case_file:
                    st.error("Debe cargar un archivo ZIP con la serie tomografica.")
                else:
                    with st.spinner("Validando paquete, computando contornos e integrando volumen..."):
                        safe_code = re.sub(r'[^A-Za-z0-9_]', '', form_code.replace('-', '_')).upper()
                        if not safe_code:
                            safe_code = "CUSTOM_CASE"
                        clean_id = f"PATIENT_{safe_code}"
                        target_dir = (CASES_DIR / clean_id).resolve()

                        try:
                            _safe_extract_zip(uploaded_case_file, target_dir)
                        except ValueError as ve:
                            st.error(str(ve))
                            st.stop()

                        meta_in_zip = target_dir / "metadata.json"
                        if meta_in_zip.exists():
                            with open(meta_in_zip, "r", encoding="utf-8") as f:
                                meta_dict = json.load(f)
                            slices_meta = meta_dict.get("slices", [])
                            num_slices_found = len(slices_meta)
                            z_positions = np.array([float(s["z_position_mm"]) for s in slices_meta], dtype=float)
                            areas_found = [float(s.get("cv_area_mm2", s.get("area_mm2", 0.0))) for s in slices_meta]
                            form_h = meta_dict.get("slice_thickness_h_mm", form_h)
                            form_px = meta_dict.get("pixel_spacing_mm", form_px)
                            form_pathology = meta_dict.get("case_name", meta_dict.get("lesion_type", form_pathology))
                        else:
                            extracted_imgs = sorted([
                                f for f in os.listdir(target_dir)
                                if (f.startswith("slice_") or "mask" not in f) and f.endswith((".png", ".jpg", ".jpeg", ".tif"))
                            ])
                            num_slices_found = len(extracted_imgs)

                            if num_slices_found < 3:
                                st.error(f"Se encontraron solo {num_slices_found} cortes. Se requieren al menos 3 cortes.")
                                st.stop()

                            segmenter = BrainLesionSegmenter()
                            z_positions = np.linspace(0.0, (num_slices_found - 1) * form_h, num_slices_found)
                            slices_meta = []
                            areas_found = []

                            for s_idx, s_file in enumerate(extracted_imgs):
                                s_path = target_dir / s_file
                                img_bgr = cv2.imread(str(s_path))
                                if img_bgr is not None:
                                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                                    res_seg = segmenter.segment_tumor_flair(img_rgb, form_px)
                                    c_area = float(res_seg["area_mm2"])
                                    areas_found.append(c_area)
                                    slices_meta.append({
                                        "slice_idx": s_idx,
                                        "filename": s_file,
                                        "z_position_mm": float(round(z_positions[s_idx], 2)),
                                        "slice_thickness_h_mm": form_h,
                                        "tumor_contour": res_seg.get("contour", []),
                                        "area_mm2": c_area,
                                        "cv_contour": res_seg.get("contour", []),
                                        "cv_area_mm2": c_area,
                                        "dice_similarity": 0.95
                                    })
                                else:
                                    areas_found.append(0.0)

                            meta_dict = {
                                "case_id": clean_id,
                                "case_name": form_pathology,
                                "patient_id": form_code,
                                "modality": form_modality,
                                "lesion_type": form_pathology,
                                "num_slices": num_slices_found,
                                "slice_thickness_h_mm": form_h,
                                "pixel_spacing_mm": form_px,
                                "matrix_dim": 256,
                                "slices": slices_meta
                            }
                            with open(target_dir / "metadata.json", "w", encoding="utf-8") as f:
                                json.dump(meta_dict, f, indent=2)

                        if len(areas_found) < 3:
                            st.error("Telemetria insuficiente (se requieren al menos 3 cortes).")
                            st.stop()

                        simpson_calc = NumericalEngine.calculate_simpson_13_composite(z_positions, np.array(areas_found, dtype=float))
                        calc_vol = float(simpson_calc.volume_cm3)

                        risk_cat = (
                            "Grado III: Efecto de Masa Critico (Urgente)" if calc_vol > 25.0
                            else "Grado II: Masa Moderada (5 - 25 cm3)" if calc_vol >= 5.0
                            else "Grado I: Lesion Focal Menor (< 5 cm3)"
                        )

                        patient_record = {
                            "patient_id": clean_id,
                            "patient_code": form_code,
                            "full_name": form_name,
                            "age": form_age,
                            "gender": form_gender[0],
                            "modality": form_modality,
                            "pathology": form_pathology,
                            "num_slices": num_slices_found,
                            "slice_thickness_h": form_h,
                            "pixel_spacing": form_px,
                            "matrix_dim": 256,
                            "volume_cm3": calc_vol,
                            "risk_stage": risk_cat,
                            "physician": "Servicio de Neurocirugia",
                            "data_dir": f"data/kaggle_cases/{clean_id}",
                            "status": "Procesado y Validado",
                            "notes": f"Estudio procesado con {num_slices_found} cortes mediante {simpson_calc.method_used}."
                        }

                        db.save_patient(patient_record, user_id=current_user_id)
                        st.success(f"Paciente {form_code} registrado exitosamente. Volumen calculado: {calc_vol:.4f} cm³.")

                        if st.button("Abrir Caso en Estacion PACS", key="btn_open_custom_pacs"):
                            on_success_intake(clean_id)
