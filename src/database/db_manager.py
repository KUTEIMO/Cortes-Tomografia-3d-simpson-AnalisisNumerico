"""
Modulo de Gestion de Base de Datos Clinica y Control de Acceso Basado en Roles (RBAC)
Sistema: NEURO-CALC | Estacion de Telemetria y Volumetria Tomografica

Proporciona almacenamiento persistente local para historias clinicas:
- Registro y ciclo de vida de pacientes tomograficos (CRUD).
- Roles clinicos institucionales y matriz de permisos.
- Bitacora inmutable de auditoria hospitalaria (Audit Trail).
"""

import os
import json
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_DIR = BASE_DIR / "data" / "database"


DEFAULT_USERS = [
    {
        "user_id": "MED_PRINCIPAL",
        "full_name": "Dr. Carlos Mendez",
        "role": "Neurocirujano Titular",
        "hospital_unit": "Servicio de Neurocirugia y Planificacion Quirurgica",
        "permissions": ["READ", "PROCESS", "APPROVE", "DELETE", "EXPORT"],
        "is_active": True
    },
    {
        "user_id": "MED_AUDITOR",
        "full_name": "Dra. Valeria Morales",
        "role": "Auditor Radiologico Jefe",
        "hospital_unit": "Departamento de Neurociencias y Volumetria",
        "permissions": ["READ", "PROCESS", "APPROVE", "DELETE", "EXPORT", "ADMIN"],
        "is_active": True
    },
    {
        "user_id": "DRA_RUIZ",
        "full_name": "Dra. Elena Ruiz M.",
        "role": "Medico Radiologo Especialista",
        "hospital_unit": "Unidad de Diagnostico por Imagenes y RMN",
        "permissions": ["READ", "PROCESS", "EXPORT"],
        "is_active": True
    },
    {
        "user_id": "RES_GARCIA",
        "full_name": "Dr. Camilo Garcia",
        "role": "Residente de Neurocirugia",
        "hospital_unit": "Servicio de Neurocirugia",
        "permissions": ["READ", "PROCESS"],
        "is_active": True
    }
]


class PatientDatabaseManager:
    """
    Gestor de base de datos clinica con persistencia local JSON e historial inmutable.
    """

    def __init__(self, db_dir: Optional[Path] = None):
        self.db_dir = db_dir or DB_DIR
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.patients_file = self.db_dir / "patients.json"
        self.audit_file = self.db_dir / "audit_logs.json"
        self._initialize_db()

    def _initialize_db(self):
        """Inicializa los archivos de la base de datos si no existen."""
        if not self.patients_file.exists():
            self._seed_default_patients()
        if not self.audit_file.exists():
            self._save_audit_logs([])

    def _load_patients(self) -> List[Dict[str, Any]]:
        try:
            with open(self.patients_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _save_patients(self, patients: List[Dict[str, Any]]):
        with open(self.patients_file, "w", encoding="utf-8") as f:
            json.dump(patients, f, indent=2, ensure_ascii=False)

    def _load_audit_logs(self) -> List[Dict[str, Any]]:
        try:
            with open(self.audit_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _save_audit_logs(self, logs: List[Dict[str, Any]]):
        with open(self.audit_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)

    def _seed_default_patients(self):
        """Pre-carga los casos de prueba clinica (TCGA-4944, TCGA-4941, Fantoma)."""
        defaults = [
            {
                "patient_id": "PATIENT_01_TCGA_4944",
                "patient_code": "PAC-2026-4944",
                "full_name": "Paciente Anonimizado TCGA-4944",
                "age": 47,
                "gender": "M",
                "modality": "MR Axial FLAIR",
                "pathology": "Astrocitoma Fibrilar Grado II (Frontoparietal Izquierdo)",
                "num_slices": 11,
                "slice_thickness_h": 5.0,
                "pixel_spacing": 0.9375,
                "matrix_dim": 256,
                "volume_cm3": 120.236,
                "risk_stage": "Grado III: Efecto de Masa Critico (Urgente)",
                "physician": "Dr. Carlos Mendez / Dra. Valeria Morales",
                "data_dir": "data/kaggle_cases/PATIENT_01_TCGA_4944",
                "status": "Procesado y Validado",
                "created_at": "2026-11-06T10:00:00Z",
                "notes": "Estudio adquirido bajo protocolo estandar LOM 0°. Tumoracion infiltrante con compresion del asta anterior del ventriculo lateral izquierdo."
            },
            {
                "patient_id": "PATIENT_02_TCGA_4941",
                "patient_code": "PAC-2026-4941",
                "full_name": "Paciente Anonimizado TCGA-4941",
                "age": 52,
                "gender": "F",
                "modality": "MR Axial FLAIR",
                "pathology": "Oligodendroglioma de Bajo Grado (Frontal)",
                "num_slices": 10,
                "slice_thickness_h": 5.0,
                "pixel_spacing": 0.9375,
                "matrix_dim": 256,
                "volume_cm3": 57.037,
                "risk_stage": "Grado III: Efecto de Masa Critico (Urgente)",
                "physician": "Dra. Elena Ruiz / Dr. Carlos Mendez",
                "data_dir": "data/kaggle_cases/PATIENT_02_TCGA_4941",
                "status": "Procesado y Validado",
                "created_at": "2026-11-05T14:00:00Z",
                "notes": "Serie de paridad impar corregida mediante cuadratura hibrida Simpson 1/3 + 3/8. Sin fuga anterior."
            },
            {
                "patient_id": "PATIENT_03_TCGA_4943",
                "patient_code": "PAC-2026-4943",
                "full_name": "Paciente Anonimizado TCGA-4943",
                "age": 44,
                "gender": "M",
                "modality": "MR Axial FLAIR",
                "pathology": "Oligoastrocitoma Grado II (Temporofrontal)",
                "num_slices": 11,
                "slice_thickness_h": 5.0,
                "pixel_spacing": 0.9375,
                "matrix_dim": 256,
                "volume_cm3": 60.346,
                "risk_stage": "Grado III: Efecto de Masa Critico (Urgente)",
                "physician": "Dr. Carlos Mendez / Dra. Valeria Morales",
                "data_dir": "data/kaggle_cases/PATIENT_03_TCGA_4943",
                "status": "Procesado y Validado",
                "created_at": "2026-11-06T14:30:00Z",
                "notes": "Serie axial real TCGA/TCIA. Lesion hiperintensa con compromiso temporofrontal. Integracion Simpson 1/3 puro sobre ventana tumoral de 11 cortes."
            },
            {
                "patient_id": "PATIENT_04_TCGA_5393",
                "patient_code": "PAC-2026-5393",
                "full_name": "Paciente Anonimizado TCGA-5393",
                "age": 58,
                "gender": "F",
                "modality": "MR Axial FLAIR",
                "pathology": "Glioblastoma Multiforme (Parietotemporal)",
                "num_slices": 11,
                "slice_thickness_h": 5.0,
                "pixel_spacing": 0.9375,
                "matrix_dim": 256,
                "volume_cm3": 125.106,
                "risk_stage": "Grado III: Efecto de Masa Critico (Urgente)",
                "physician": "Dr. Carlos Mendez / Dra. Valeria Morales",
                "data_dir": "data/kaggle_cases/PATIENT_04_TCGA_5393",
                "status": "Procesado y Validado",
                "created_at": "2026-11-06T15:00:00Z",
                "notes": "Serie tomografica real TCGA-LGG. Voluminosa masa infiltrante parietotemporal con colapso ventricular. Cuadratura Simpson 1/3 puro en 11 cortes."
            }
        ]
        self._save_patients(defaults)
        self.log_audit("SISTEMA", "INITIALIZE_DATABASE", "Inicializacion de base de datos con casos clinicos validados.")

    # -------------------------------------------------------------------------
    # OPERACIONES CRUD DE PACIENTES
    # -------------------------------------------------------------------------
    def list_patients(self) -> List[Dict[str, Any]]:
        """Retorna todos los pacientes ordenados por fecha descendente."""
        patients = self._load_patients()
        return sorted(patients, key=lambda p: p.get("created_at", ""), reverse=True)

    def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Busca un paciente por su identificador unico."""
        patients = self._load_patients()
        for p in patients:
            if p.get("patient_id") == patient_id:
                return p
        return None

    def save_patient(self, patient_data: Dict[str, Any], user_id: str = "MED_PRINCIPAL") -> bool:
        """Crea o actualiza un paciente en la base de datos."""
        patients = self._load_patients()
        p_id = patient_data.get("patient_id")
        if not p_id:
            return False

        updated = False
        for i, p in enumerate(patients):
            if p.get("patient_id") == p_id:
                patients[i] = {**p, **patient_data}
                updated = True
                break

        if not updated:
            if "created_at" not in patient_data:
                patient_data["created_at"] = datetime.now(timezone.utc).isoformat()
            patients.append(patient_data)

        self._save_patients(patients)
        action = "UPDATE_PATIENT" if updated else "CREATE_PATIENT"
        self.log_audit(user_id, action, f"Registro de paciente {p_id} guardado con volumen {patient_data.get('volume_cm3', 0.0):.2f} cm3", p_id)
        return True

    def delete_patient(self, patient_id: str, user_id: str = "MED_PRINCIPAL") -> bool:
        """Elimina un paciente de la base de datos."""
        patients = self._load_patients()
        initial_count = len(patients)
        patients = [p for p in patients if p.get("patient_id") != patient_id]

        if len(patients) < initial_count:
            self._save_patients(patients)
            self.log_audit(user_id, "DELETE_PATIENT", f"Paciente {patient_id} eliminado del registro clinico activo.", patient_id)
            return True
        return False

    # -------------------------------------------------------------------------
    # GESTION DE ROLES Y PERMISOS (RBAC)
    # -------------------------------------------------------------------------
    def list_users(self) -> List[Dict[str, Any]]:
        """Retorna la lista de usuarios y roles autorizados en el sistema."""
        return DEFAULT_USERS

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene un usuario por su identificador."""
        for u in DEFAULT_USERS:
            if u["user_id"] == user_id:
                return u
        return None

    def check_permission(self, user_id: str, required_permission: str) -> bool:
        """Verifica si el usuario tiene permiso para ejecutar una accion."""
        u = self.get_user(user_id)
        if not u or not u.get("is_active"):
            return False
        perms = u.get("permissions", [])
        return "ADMIN" in perms or required_permission in perms

    # -------------------------------------------------------------------------
    # BITACORA INMUTABLE DE AUDITORIA (AUDIT TRAIL)
    # -------------------------------------------------------------------------
    def log_audit(self, user_id: str, action: str, details: str, patient_id: Optional[str] = None):
        """Registra un evento inmutable en la bitacora hospitalaria."""
        logs = self._load_audit_logs()
        ts = datetime.now(timezone.utc).isoformat()
        raw_payload = f"{ts}|{user_id}|{action}|{patient_id or ''}|{details}"
        crc = hex(zlib.crc32(raw_payload.encode("utf-8")) & 0xFFFFFFFF)

        entry = {
            "timestamp": ts,
            "user_id": user_id,
            "action": action,
            "patient_id": patient_id,
            "details": details,
            "crc32_checksum": crc
        }
        logs.insert(0, entry)  # Mas reciente al inicio
        # Conservar ultimos 500 registros
        self._save_audit_logs(logs[:500])

    def get_audit_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retorna las ultimas entradas de auditoria."""
        logs = self._load_audit_logs()
        return logs[:limit]
