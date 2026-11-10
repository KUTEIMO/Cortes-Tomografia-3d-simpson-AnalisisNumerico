"""
Modulo de Control de Acceso Basado en Roles y Auditoria (RBAC) - NEURO-CALC
"""

from typing import Callable
import pandas as pd
import streamlit as st
from src.database.db_manager import PatientDatabaseManager


def render_roles_and_audit(
    db: PatientDatabaseManager,
    current_user_id: str,
    on_change_user: Callable[[str], None],
    embedded: bool = False
):
    """Renderiza la interfaz de gestion de roles y bitacora de auditoria."""
    if not embedded:
        st.markdown("""
        <div class="page-header-wrap">
            <div class="page-eyebrow">Seguridad Institucional</div>
            <h1 class="page-title">Control de acceso y auditoria</h1>
            <p class="page-subtitle">Gestion de perfiles facultativos, matriz de privilegios y registro inmutable de operaciones.</p>
        </div>
        """, unsafe_allow_html=True)

    users = db.list_users()
    active_user = db.get_user(current_user_id) or users[0]

    # Tarjeta unificada de usuario activo y selector
    with st.container(border=True):
        col_u1, col_u2 = st.columns([3.0, 2.0])

        with col_u1:
            st.markdown(f"""
            <div style="font-size: 0.72rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em;">Facultativo en sesion</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #0F172A; margin-top: 2px;">{active_user['full_name']}</div>
            <div style="font-size: 0.84rem; color: #475469; margin-top: 2px;">
                <strong>Rol:</strong> {active_user['role']} &nbsp;&middot;&nbsp; <strong>Unidad:</strong> {active_user['hospital_unit']}
            </div>
            <div style="margin-top: 8px;">
                {' '.join([f'<span class=\"status-pill pill-stable\" style=\"margin-right: 4px;\">{p}</span>' for p in active_user['permissions']])}
            </div>
            """, unsafe_allow_html=True)

        with col_u2:
            st.markdown("<div style='font-size: 0.76rem; font-weight: 600; color: #344054; margin-bottom: 6px;'>Conmutar Perfil de Usuario</div>", unsafe_allow_html=True)
            user_options = {u["user_id"]: f"{u['full_name']} ({u['role']})" for u in users}
            selected_uid = st.selectbox(
                "Seleccionar Usuario",
                options=list(user_options.keys()),
                format_func=lambda x: user_options[x],
                index=list(user_options.keys()).index(current_user_id) if current_user_id in user_options else 0,
                label_visibility="collapsed"
            )
            if selected_uid != current_user_id:
                on_change_user(selected_uid)
                st.rerun()

    tab_matrix, tab_trail = st.tabs([
        "Matriz de Permisos (RBAC)",
        "Bitacora de Auditoria (CRC32)"
    ])

    with tab_matrix:
        with st.container(border=True):
            st.markdown("<div style='font-weight: 600; color: #101828; margin-bottom: 8px;'>Privilegios por Perfil Clinico</div>", unsafe_allow_html=True)

            matrix_rows = []
            for u in users:
                perms = u.get("permissions", [])
                matrix_rows.append({
                    "Facultativo": u["full_name"],
                    "Rol Clinico": u["role"],
                    "Lectura PACS": "SI" if "READ" in perms else "NO",
                    "Segmentacion": "SI" if "PROCESS" in perms else "NO",
                    "Aprobacion": "SI" if "APPROVE" in perms else "NO",
                    "Baja de Casos": "SI" if "DELETE" in perms or "ADMIN" in perms else "NO",
                    "Exportacion": "SI" if "EXPORT" in perms else "NO",
                    "Administracion": "SI" if "ADMIN" in perms else "NO"
                })
            st.dataframe(pd.DataFrame(matrix_rows), width="stretch", hide_index=True)

    with tab_trail:
        with st.container(border=True):
            st.markdown("<div style='font-weight: 600; color: #101828; margin-bottom: 4px;'>Registro Inmutable de Transacciones</div>", unsafe_allow_html=True)
            st.markdown("<div style='font-size: 0.78rem; color: #475467; margin-bottom: 10px;'>Cada operacion genera una marca temporal ISO 8601 y suma de comprobacion CRC32:</div>", unsafe_allow_html=True)

            logs = db.get_audit_logs(limit=50)
            log_rows = []
            for l in logs:
                log_rows.append({
                    "Fecha / Hora (UTC)": l.get("timestamp", "")[:19].replace("T", " "),
                    "Usuario": l.get("user_id", ""),
                    "Operacion": l.get("action", ""),
                    "Caso": l.get("patient_id") or "SISTEMA",
                    "Detalle": l.get("details", ""),
                    "CRC32": l.get("crc32_checksum", "")
                })

            if log_rows:
                st.dataframe(pd.DataFrame(log_rows), width="stretch", hide_index=True)
            else:
                st.markdown("<div style='color: #667085;'>No hay registros disponibles.</div>", unsafe_allow_html=True)
