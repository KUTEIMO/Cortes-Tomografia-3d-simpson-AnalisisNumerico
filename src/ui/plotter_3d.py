"""
Modulo de Renderizado Grafico 3D y Perfiles Axiales (Plotly)

Visualizaciones cientificas modulares inspiradas en Braintrust / Mobbin:
1. Reconstruccion 3D anatomica completa del cerebro humano (corteza translucida + lesion interna).
2. Plano de corte interactivo sincronizado con el visor 2D.
3. Perfil seccional 2D con spline cubico y region de integracion de Simpson.
4. Grafico comparativo de error de cuadraturas.
"""

from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import plotly.graph_objects as go
from src.engine.spline_interpolator import NaturalCubicSpline
from src.engine.error_evaluator import BenchmarkComparison


# Tokens de color clinico
THEME_BG = "#FFFFFF"       # Fondo de Tarjetas (Cards)
THEME_PANEL = "#FFFFFF"    # Superficie interna
THEME_GRID = "#EAECF0"     # Bordes y grillas sutiles
THEME_TEXT = "#101828"     # Texto Principal
THEME_MUTED = "#475467"    # Texto Secundario / Labels
THEME_BLUE = "#2563EB"     # Acento Primario (Cobalt Blue)
THEME_AMBER = "#F59E0B"    # Acento Secundario (Amber)
THEME_SUCCESS = "#027A48"  # Acento Exito
THEME_CRIMSON = "#D92D20"  # Error critico


def generate_brain_cortex_coordinates(n_u: int = 45, n_v: int = 35) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Genera la geometria tridimensional de la corteza cerebral humana:
    - Lobulos frontal, parietal, temporal y occipital.
    - Cisura interhemisferica longitudinal medial.
    - Convoluciones y surcos corticales continuos.
    Dimensiones MNI estandar: X ~ 136 mm, Y ~ 175 mm, Z ~ 115 mm.
    """
    u = np.linspace(0.05, np.pi - 0.05, n_u)
    v = np.linspace(0, 2 * np.pi, n_v)
    u_grid, v_grid = np.meshgrid(u, v)

    rx, ry, rz = 66.0, 84.0, 56.0
    x = rx * np.sin(u_grid) * np.cos(v_grid)
    y = ry * np.sin(u_grid) * np.sin(v_grid)
    z = rz * np.cos(u_grid)

    # 1. Cisura interhemisferica (separacion medial entre hemisferios)
    fissure = 1.0 - 0.22 * np.exp(-(x / 12.0)**2) * (z > -15.0)
    # 2. Asimetria y afinamiento frontal vs ensanchamiento parietal
    taper = 1.0 + 0.14 * (y / ry) * (z > 0)
    x = x * fissure
    y = y * taper

    # 3. Fosas temporales bilaterales y fosa posterior cerebelosa
    temporal_indent = 1.0 - 0.12 * np.exp(-((x**2) / 600.0 + ((y + 15.0)**2) / 900.0)) * (z < 0)
    x = x * temporal_indent

    # 4. Giros y surcos corticales (micro-ondulaciones anatomicas)
    gyri = 1.4 * np.sin(0.32 * x) * np.cos(0.32 * y) * np.sin(0.32 * z)
    x += gyri * np.sin(u_grid) * np.cos(v_grid)
    y += gyri * np.sin(u_grid) * np.sin(v_grid)
    z += gyri * np.cos(u_grid)

    return x, y, z


from scipy.spatial import Delaunay


def resample_by_perimeter(pts: np.ndarray, m_resample: int = 72) -> Optional[np.ndarray]:
    """
    Remuestrea un contorno poligonal arbitrario a lo largo de su perimetro acumulado (abscisa curvilinea s in [0, 1]).
    A diferencia del ordenamiento polar, este metodo preserva al 100% todas las concavidades,
    depresiones anatomicas, muescas y lobulaciones del tumor biologico sin colapsar pliegues.
    """
    pts = np.array(pts, dtype=float)
    if len(pts) < 4:
        return None
    pts_closed = np.vstack([pts, pts[0]])
    dists = np.sqrt(np.sum(np.diff(pts_closed, axis=0)**2, axis=1))
    cum_dist = np.insert(np.cumsum(dists), 0, 0)
    total_L = cum_dist[-1]
    if total_L == 0:
        return None
    norm_s = cum_dist / total_L
    target_s = np.linspace(0, 1, m_resample, endpoint=False)
    rx = np.interp(target_s, norm_s, pts_closed[:, 0])
    ry = np.interp(target_s, norm_s, pts_closed[:, 1])
    return np.column_stack([rx, ry])


def build_organic_surface_mesh(
    slices_data: List[Dict[str, Any]],
    contour_key: str = "tumor_contour",
    pixel_spacing_mm: float = 0.9375,
    matrix_dim: int = 256,
    h_mm: float = 5.0,
    m_resample: int = 72,
    color: str = "#2563EB",
    opacity: float = 0.92,
    name: str = "Masa Tumoral 3D",
    is_cranium: bool = False
) -> Tuple[Optional[go.Mesh3d], List[go.Scatter3d]]:
    """
    Construye una malla tridimensional cerrada y suave a partir de poligonos reales
    mediante lofting por parametrizacion de perimetro y alineacion de fase angular.
    Si is_cranium=True, interpola y cierra la boveda craneal completa (calvaria y base)
    eliminando el aspecto tubular/cilindrico y reproduciendo la curvatura anatomica real.
    """
    center_px = matrix_dim / 2.0
    valid_slices = [s for s in slices_data if len(s.get(contour_key, [])) >= 4]

    if len(valid_slices) < 2:
        return None, []

    resampled_layers = []
    z_levels = []
    contour_traces = []

    for idx, s in enumerate(valid_slices):
        raw_pts = np.array(s[contour_key], dtype=float)
        r_pts = resample_by_perimeter(raw_pts, m_resample=m_resample)
        if r_pts is not None:
            resampled_layers.append(r_pts)
            z_val = float(s["z_position_mm"])
            z_levels.append(z_val)

            # Traza de contorno exacto sin remuestreo (Scatter3d de maxima fidelidad)
            orig_x_mm = (raw_pts[:, 0] - center_px) * pixel_spacing_mm
            orig_y_mm = -(raw_pts[:, 1] - center_px) * pixel_spacing_mm
            orig_z_mm = np.full(len(raw_pts), z_val)
            orig_x_mm = np.append(orig_x_mm, orig_x_mm[0])
            orig_y_mm = np.append(orig_y_mm, orig_y_mm[0])
            orig_z_mm = np.append(orig_z_mm, orig_z_mm[0])

            contour_traces.append(
                go.Scatter3d(
                    x=orig_x_mm,
                    y=orig_y_mm,
                    z=orig_z_mm,
                    mode="lines",
                    line=dict(color="#101828" if "tumor" in contour_key else "#64748B", width=2.5),
                    name=f"Contorno Corte {s.get('slice_index', idx)} (z={z_val:.1f}mm)",
                    hovertemplate=f"Corte {s.get('slice_index', idx)}<br>z = {z_val:.2f} mm<br>Area = {s.get('area_mm2', 0.0):.1f} mm²<extra></extra>"
                )
            )

    K = len(resampled_layers)
    if K < 2:
        return None, contour_traces

    # Alinear fase entre capas adyacentes para prevenir torsion en la malla
    aligned_layers = [resampled_layers[0]]
    for k in range(1, K):
        prev = aligned_layers[-1]
        curr = resampled_layers[k]
        dists = np.sum((curr - prev[0])**2, axis=1)
        best_shift = int(np.argmin(dists))
        aligned_curr = np.roll(curr, -best_shift, axis=0)
        aligned_layers.append(aligned_curr)

    all_x, all_y, all_z = [], [], []
    for k, layer in enumerate(aligned_layers):
        x_mm = (layer[:, 0] - center_px) * pixel_spacing_mm
        y_mm = -(layer[:, 1] - center_px) * pixel_spacing_mm
        z_mm = np.full(m_resample, z_levels[k])
        all_x.extend(x_mm.tolist())
        all_y.extend(y_mm.tolist())
        all_z.extend(z_mm.tolist())

    tri_i, tri_j, tri_k = [], [], []

    # Triangulacion de la cinta lateral entre cortes contiguos
    for k in range(K - 1):
        for m in range(m_resample):
            mn = (m + 1) % m_resample
            v00 = k * m_resample + m
            v01 = k * m_resample + mn
            v10 = (k + 1) * m_resample + m
            v11 = (k + 1) * m_resample + mn
            tri_i.extend([v00, v00])
            tri_j.extend([v10, v11])
            tri_k.extend([v11, v01])

    if is_cranium:
        # Cierre anatomico de la boveda craneal (calvaria superior y base de craneo)
        top_cent = aligned_layers[-1].mean(axis=0)
        top_pts = aligned_layers[-1]
        dome_ring_pts = top_cent + (top_pts - top_cent) * 0.58
        z_dome = z_levels[-1] + 7.0

        dome_start = len(all_x)
        for pt in dome_ring_pts:
            all_x.append((pt[0] - center_px) * pixel_spacing_mm)
            all_y.append(-(pt[1] - center_px) * pixel_spacing_mm)
            all_z.append(z_dome)

        top_apex_idx = len(all_x)
        all_x.append((top_cent[0] - center_px) * pixel_spacing_mm)
        all_y.append(-(top_cent[1] - center_px) * pixel_spacing_mm)
        all_z.append(z_levels[-1] + 13.5)

        bot_cent = aligned_layers[0].mean(axis=0)
        bot_pts = aligned_layers[0]
        base_ring_pts = bot_cent + (bot_pts - bot_cent) * 0.65
        z_base_ring = z_levels[0] - 6.5

        base_start = len(all_x)
        for pt in base_ring_pts:
            all_x.append((pt[0] - center_px) * pixel_spacing_mm)
            all_y.append(-(pt[1] - center_px) * pixel_spacing_mm)
            all_z.append(z_base_ring)

        bot_apex_idx = len(all_x)
        all_x.append((bot_cent[0] - center_px) * pixel_spacing_mm)
        all_y.append(-(bot_cent[1] - center_px) * pixel_spacing_mm)
        all_z.append(z_levels[0] - 12.0)

        # Triangulacion de la cupula superior: capa K-1 -> dome_start -> top_apex
        top_layer_start = (K - 1) * m_resample
        for m in range(m_resample):
            mn = (m + 1) % m_resample
            v00 = top_layer_start + m
            v01 = top_layer_start + mn
            v10 = dome_start + m
            v11 = dome_start + mn
            tri_i.extend([v00, v00])
            tri_j.extend([v10, v11])
            tri_k.extend([v11, v01])

            # Cono hacia el apice superior
            tri_i.append(top_apex_idx)
            tri_j.append(dome_start + m)
            tri_k.append(dome_start + mn)

        # Triangulacion de la base inferior: capa 0 -> base_start -> bot_apex
        bot_layer_start = 0
        for m in range(m_resample):
            mn = (m + 1) % m_resample
            v00 = bot_layer_start + m
            v01 = bot_layer_start + mn
            v10 = base_start + m
            v11 = base_start + mn
            tri_i.extend([v00, v00])
            tri_j.extend([v01, v10])
            tri_k.extend([v10, v11])

            # Cono hacia el apice inferior
            tri_i.append(bot_apex_idx)
            tri_j.append(base_start + mn)
            tri_k.append(base_start + m)

    else:
        # Tapa inferior para masa tumoral
        bot_cent = aligned_layers[0].mean(axis=0)
        all_x.append((bot_cent[0] - center_px) * pixel_spacing_mm)
        all_y.append(-(bot_cent[1] - center_px) * pixel_spacing_mm)
        all_z.append(z_levels[0] - h_mm * 0.45)
        bot_idx = len(all_x) - 1

        # Tapa superior para masa tumoral
        top_cent = aligned_layers[-1].mean(axis=0)
        all_x.append((top_cent[0] - center_px) * pixel_spacing_mm)
        all_y.append(-(top_cent[1] - center_px) * pixel_spacing_mm)
        all_z.append(z_levels[-1] + h_mm * 0.45)
        top_idx = len(all_x) - 1

        for m in range(m_resample):
            mn = (m + 1) % m_resample
            tri_i.append(bot_idx)
            tri_j.append(mn)
            tri_k.append(m)

        for m in range(m_resample):
            mn = (m + 1) % m_resample
            tri_i.append(top_idx)
            tri_j.append((K - 1) * m_resample + m)
            tri_k.append((K - 1) * m_resample + mn)

    lighting_dict = (
        dict(ambient=0.55, diffuse=0.85, roughness=0.35, specular=0.45)
        if is_cranium
        else dict(ambient=0.45, diffuse=0.85, roughness=0.25, specular=0.55)
    )

    mesh = go.Mesh3d(
        x=all_x,
        y=all_y,
        z=all_z,
        i=tri_i,
        j=tri_j,
        k=tri_k,
        color=color,
        opacity=opacity,
        name=name,
        flatshading=False,
        lighting=lighting_dict,
        hovertemplate=f"{name}<br>Morfologia biologica pura<br>Simpson 1/3<extra></extra>"
    )

    return mesh, contour_traces


def create_3d_full_brain_visualization(
    z_coords: np.ndarray,
    areas: np.ndarray,
    active_z: Optional[float] = None,
    contour_slices: Optional[List[Dict[str, Any]]] = None,
    full_brain_slices: Optional[List[Dict[str, Any]]] = None,
    pixel_spacing_mm: float = 0.9375,
    matrix_dim: int = 256,
    h_mm: float = 5.0,
    single_slice_idx: Optional[int] = None,
    lesion_color: str = "#2563EB",
    view_mode: str = "Contorno Clinico (Cobalto)"
) -> go.Figure:
    """
    Renderiza la reconstruccion 3D de alta resolucion clinica:
    - Modo A (Volumen Completo): Reconstruccion lofted continua organica de la masa tumoral
      anidada dentro de la boveda craneal completa del paciente (calvaria superior y base).
    - Modo B (Solo Corte Activo): Renderiza UNICAMENTE la capa poligonal planar 1:1 del corte seleccionado,
      permitiendo verificar al milimetro que la forma 3D coincide al 100% con la RMN 2D en contorno y orientacion.
    """
    fig = go.Figure()
    center_px = matrix_dim / 2.0

    # Determinar que contorno usar segun el modo de inspeccion
    use_cv = "Vision Artificial" in view_mode
    is_dual = "Auditoria Dual" in view_mode
    contour_primary_key = "cv_contour" if use_cv else "tumor_contour"
    primary_color = "#059669" if use_cv else lesion_color

    # MODO B: INSPECCION 1:1 DE UN SOLO CORTE SELECCIONADO
    if single_slice_idx is not None and contour_slices and 0 <= single_slice_idx < len(contour_slices):
        cur_s = contour_slices[single_slice_idx]
        cur_z = float(cur_s["z_position_mm"])
        raw_tumor = np.array(cur_s.get(contour_primary_key) or cur_s.get("tumor_contour", []), dtype=float)
        raw_brain = np.array(cur_s.get("brain_contour", []), dtype=float)

        # 1. Contorno y superficie plana de la lesion
        if len(raw_tumor) >= 4:
            x_mm = (raw_tumor[:, 0] - center_px) * pixel_spacing_mm
            y_mm = -(raw_tumor[:, 1] - center_px) * pixel_spacing_mm
            z_mm = np.full(len(raw_tumor), cur_z)

            # Triangulacion planar 2D del poligono interior
            try:
                tri = Delaunay(raw_tumor)
                valid_simplices = []
                for sim in tri.simplices:
                    cent = raw_tumor[sim].mean(axis=0)
                    if cv2.pointPolygonTest(raw_tumor.astype(np.int32), (float(cent[0]), float(cent[1])), False) >= 0:
                        valid_simplices.append(sim)
                if valid_simplices:
                    valid_s = np.array(valid_simplices)
                    fig.add_trace(
                        go.Mesh3d(
                            x=x_mm, y=y_mm, z=z_mm,
                            i=valid_s[:, 0], j=valid_s[:, 1], k=valid_s[:, 2],
                            color=primary_color,
                            opacity=0.92,
                            name=f"Seccion Tumoral Corte {single_slice_idx + 1} (1:1 RMN)",
                            lighting=dict(ambient=0.5, diffuse=0.8)
                        )
                    )
            except Exception:
                pass

            # Linea de contorno vectorial gruesa (Scatter3d)
            x_line = np.append(x_mm, x_mm[0])
            y_line = np.append(y_mm, y_mm[0])
            z_line = np.append(z_mm, z_mm[0])
            c_name = f"Contorno IA Corte {single_slice_idx + 1}" if use_cv else f"Contorno Real Corte {single_slice_idx + 1}"
            fig.add_trace(
                go.Scatter3d(
                    x=x_line, y=y_line, z=z_line,
                    mode="lines",
                    line=dict(color=primary_color, width=5.0),
                    name=c_name
                )
            )

        # En modo Auditoria Dual, anadir tambien el contorno alternativo
        if is_dual and "cv_contour" in cur_s and len(cur_s.get("cv_contour", [])) >= 4:
            cv_raw = np.array(cur_s["cv_contour"], dtype=float)
            cv_x = (cv_raw[:, 0] - center_px) * pixel_spacing_mm
            cv_y = -(cv_raw[:, 1] - center_px) * pixel_spacing_mm
            cv_z = np.full(len(cv_raw), cur_z)
            fig.add_trace(
                go.Scatter3d(
                    x=np.append(cv_x, cv_x[0]),
                    y=np.append(cv_y, cv_y[0]),
                    z=np.append(cv_z, cv_z[0]),
                    mode="lines",
                    line=dict(color="#059669", width=4.5, dash="dash"),
                    name="Vision Artificial IA (Refinada)"
                )
            )

        # 2. Perimetro craneal del corte
        if len(raw_brain) >= 4:
            bx_mm = (raw_brain[:, 0] - center_px) * pixel_spacing_mm
            by_mm = -(raw_brain[:, 1] - center_px) * pixel_spacing_mm
            bz_mm = np.full(len(raw_brain), cur_z)
            bx_line = np.append(bx_mm, bx_mm[0])
            by_line = np.append(by_mm, by_mm[0])
            bz_line = np.append(bz_mm, bz_mm[0])
            fig.add_trace(
                go.Scatter3d(
                    x=bx_line, y=by_line, z=bz_line,
                    mode="lines",
                    line=dict(color="#94A3B8", width=2.5, dash="dash"),
                    name="Perimetro Craneal del Corte"
                )
            )

        # 3. Plano axial del corte activo
        px = np.array([-90.0, 90.0, 90.0, -90.0])
        py = np.array([-110.0, -110.0, 110.0, 110.0])
        pz = np.full(4, cur_z)
        fig.add_trace(
            go.Mesh3d(
                x=px, y=py, z=pz,
                i=[0, 0], j=[1, 2], k=[2, 3],
                color="#0EA5E9", opacity=0.18,
                name=f"Plano Axial Activo (z = {cur_z:.1f} mm)"
            )
        )

        # 4. Marcadores cardinales anatomicos en el plano 3D (orientacion 1:1 identica al corte 2D)
        fig.add_trace(
            go.Scatter3d(
                x=[0, 0, -100, 100],
                y=[100, -100, 0, 0],
                z=[cur_z, cur_z, cur_z, cur_z],
                mode="text",
                text=["ANTERIOR (A)", "POSTERIOR (P)", "DERECHA (R)", "IZQUIERDA (L)"],
                textfont=dict(color="#2563EB", size=10, family="Inter"),
                name="Orientacion Cardinal Anatomica",
                hoverinfo="skip"
            )
        )

    # MODO A: RECONSTRUCCION COMPLETA DEL VOLUMEN ACUMULADO
    else:
        # 1. Capa Cortical Cerebral Externa Real del Paciente (Serie Craneal Completa)
        brain_mesh = None
        brain_source = full_brain_slices if full_brain_slices else contour_slices
        if brain_source:
            brain_mesh, _ = build_organic_surface_mesh(
                slices_data=brain_source,
                contour_key="brain_contour",
                pixel_spacing_mm=pixel_spacing_mm,
                matrix_dim=matrix_dim,
                h_mm=h_mm,
                m_resample=64,
                color="#94A3B8",
                opacity=0.16,
                name="Boveda Craneal Real (Paciente)",
                is_cranium=True
            )

        if brain_mesh is not None:
            fig.add_trace(brain_mesh)
        else:
            bx, by, bz = generate_brain_cortex_coordinates()
            fig.add_trace(
                go.Surface(
                    x=bx, y=by, z=bz,
                    colorscale=[[0, "#CBD5E1"], [0.5, "#94A3B8"], [1, "#E2E8F0"]],
                    showscale=False, opacity=0.15,
                    name="Corteza Cerebral MNI", hoverinfo="skip"
                )
            )

        # 2. Capa Interna de la Lesion / Tumor Organico
        tumor_mesh = None
        contour_traces = []

        if contour_slices:
            t_mesh_name = "Masa Tumoral IA (Refinada 3D)" if use_cv else "Masa Tumoral Real 3D"
            tumor_mesh, contour_traces = build_organic_surface_mesh(
                slices_data=contour_slices,
                contour_key=contour_primary_key,
                pixel_spacing_mm=pixel_spacing_mm,
                matrix_dim=matrix_dim,
                h_mm=h_mm,
                m_resample=72,
                color=primary_color,
                opacity=0.92,
                name=t_mesh_name,
                is_cranium=False
            )

        if tumor_mesh is not None:
            fig.add_trace(tumor_mesh)
            for tr in contour_traces:
                fig.add_trace(tr)
        else:
            radii = np.sqrt(np.maximum(0.0, areas) / np.pi)
            if len(z_coords) >= 3 and np.max(radii) > 0.1:
                spline = NaturalCubicSpline(z_coords, radii)
                z_dense = np.linspace(z_coords[0], z_coords[-1], 35)
                r_dense = spline.evaluate(z_dense)
                theta = np.linspace(0, 2 * np.pi, 28)
                theta_grid, z_grid = np.meshgrid(theta, z_dense)
                r_grid = np.tile(r_dense[:, np.newaxis], (1, len(theta)))
                fig.add_trace(
                    go.Surface(
                        x=r_grid * np.cos(theta_grid),
                        y=r_grid * np.sin(theta_grid),
                        z=z_grid,
                        colorscale=[[0, "#1D4ED8"], [0.5, "#2563EB"], [1, "#60A5FA"]],
                        showscale=False, opacity=0.92,
                        name="Masa Esferica (Calibracion)"
                    )
                )

        # 3. Capa del Plano de Corte Axial Activo
        if active_z is not None:
            px = np.array([-90.0, 90.0, 90.0, -90.0])
            py = np.array([-110.0, -110.0, 110.0, 110.0])
            pz = np.full(4, active_z)
            fig.add_trace(
                go.Mesh3d(
                    x=px, y=py, z=pz,
                    i=[0, 0], j=[1, 2], k=[2, 3],
                    color="#0EA5E9", opacity=0.22,
                    name=f"Plano Axial Activo (z = {active_z:.2f} mm)",
                    hoverinfo="skip"
                )
            )

    # Camara optimizada para coincidir al 100% con la orientacion axial de la RMN 2D
    # (Anterior arriba, Posterior abajo, Izquierda derecha, Derecha izquierda)
    cam_aligned_top = dict(eye=dict(x=0.0, y=0.25, z=2.40), up=dict(x=0.0, y=1.0, z=0.0))
    cam_perspective = dict(eye=dict(x=1.75, y=1.75, z=1.20), up=dict(x=0.0, y=0.0, z=1.0))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X Sagital (mm)", backgroundcolor="#FFFFFF", gridcolor=THEME_GRID, color=THEME_MUTED, zerolinecolor=THEME_GRID),
            yaxis=dict(title="Y Coronal (mm)", backgroundcolor="#FFFFFF", gridcolor=THEME_GRID, color=THEME_MUTED, zerolinecolor=THEME_GRID),
            zaxis=dict(title="Z Axial (mm)", backgroundcolor="#FFFFFF", gridcolor=THEME_GRID, color=THEME_MUTED, zerolinecolor=THEME_GRID),
            bgcolor=THEME_BG,
            camera=cam_aligned_top
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.0,
                y=1.08,
                xanchor="left",
                yanchor="top",
                buttons=[
                    dict(
                        label="Vista Superior (Alinear 1:1 con RMN)",
                        method="relayout",
                        args=[{"scene.camera": cam_aligned_top}]
                    ),
                    dict(
                        label="Perspectiva 3D",
                        method="relayout",
                        args=[{"scene.camera": cam_perspective}]
                    )
                ],
                font=dict(size=10, family="Inter", color=THEME_TEXT),
                bgcolor="#FFFFFF",
                bordercolor=THEME_GRID
            )
        ],
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor=THEME_BG,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.12,
            xanchor="center",
            x=0.5,
            font=dict(size=10, family="Inter", color=THEME_MUTED)
        ),
        font=dict(family="Inter, sans-serif", color=THEME_TEXT)
    )
    return fig


def create_2d_slice_view(
    img_rgb: np.ndarray,
    contour_pts: Optional[List[List[float]]] = None,
    cv_contour_pts: Optional[List[List[float]]] = None,
    pixel_spacing_mm: float = 0.9375,
    matrix_dim: int = 256,
    active_z: float = 0.0,
    slice_idx: int = 0,
    total_slices: int = 20,
    area_mm2: float = 0.0,
    cv_area_mm2: float = 0.0,
    dice_score: float = 1.0,
    fn_mm2: float = 0.0,
    fp_mm2: float = 0.0,
    view_mode: str = "Contorno Clinico (Cobalto)"
) -> go.Figure:
    """
    Renderiza el visor bidimensional de resonancia magnetica con coordenadas milimetricas
    reales en los ejes X Sagital e Y Coronal, etiquetas anatomicas y overlay vectorial.
    Mapeo espacial calibrado:
    - Eje X: [-half_w, +half_w] mm (Derecha radiologica a la izquierda, Izquierda a la derecha).
    - Eje Y: [-half_h, +half_h] mm (Posterior/Occipucio abajo, Anterior/Cara/Ojos arriba).
    """
    h_img, w_img = img_rgb.shape[:2]
    center_px = matrix_dim / 2.0
    half_w = (w_img / 2.0) * pixel_spacing_mm
    half_h = (h_img / 2.0) * pixel_spacing_mm

    fig = go.Figure()

    # Imagen Medica con calibracion espacial milimetrica exacta
    fig.add_trace(
        go.Image(
            z=img_rgb,
            x0=-half_w,
            dx=pixel_spacing_mm,
            y0=half_h,
            dy=-pixel_spacing_mm,
            hovertemplate="X Sagital: %{x:.1f} mm<br>Y Coronal: %{y:.1f} mm<extra></extra>"
        )
    )

    def to_mm_poly(pts_list):
        arr = np.array(pts_list, dtype=float)
        x = (arr[:, 0] - center_px) * pixel_spacing_mm
        y = -(arr[:, 1] - center_px) * pixel_spacing_mm
        return np.append(x, x[0]), np.append(y, y[0])

    # 1. Modo Contorno Clinico Puro (Ground Truth del Radiologo en Cobalto)
    if view_mode == "Contorno Clinico (Cobalto)" and contour_pts and len(contour_pts) >= 4:
        cx_mm, cy_mm = to_mm_poly(contour_pts)
        fig.add_trace(
            go.Scatter(
                x=cx_mm, y=cy_mm,
                mode="lines",
                line=dict(color="#2563EB", width=2.8),
                name="Mascara Radiologo (GT)",
                hovertemplate=f"Perimetro Clinico TCGA<br>Area = {area_mm2:.1f} mm²<extra></extra>"
            )
        )
        cent_x_mm = float(cx_mm[:-1].mean())
        cent_y_mm = float(cy_mm[:-1].mean())
        fig.add_trace(
            go.Scatter(
                x=[cent_x_mm], y=[cent_y_mm],
                mode="markers",
                marker=dict(color="#F59E0B", size=8, symbol="circle"),
                name="Centroide",
                hovertemplate=f"Centroide Anatomico<br>X = {cent_x_mm:.1f} mm, Y = {cent_y_mm:.1f} mm<extra></extra>"
            )
        )
        fig.add_annotation(
            x=min(cx_mm), y=max(cy_mm) + 6,
            text=f"A = {area_mm2:.1f} mm²",
            showarrow=False,
            font=dict(size=12, color="#2563EB", family="Inter"),
            bgcolor="rgba(255, 255, 255, 0.85)",
            bordercolor="#2563EB",
            borderwidth=1,
            borderpad=3
        )

    # 2. Modo Vision Artificial Refinada (Segmentacion Espectral Adaptativa en Esmeralda)
    elif view_mode == "Vision Artificial Refinada (Esmeralda)":
        active_pts = cv_contour_pts if (cv_contour_pts and len(cv_contour_pts) >= 4) else contour_pts
        active_area = cv_area_mm2 if cv_area_mm2 > 0 else area_mm2
        if active_pts and len(active_pts) >= 4:
            cx_mm, cy_mm = to_mm_poly(active_pts)
            fig.add_trace(
                go.Scatter(
                    x=cx_mm, y=cy_mm,
                    mode="lines",
                    line=dict(color="#059669", width=2.8),
                    name="Vision Artificial (IA Refinada)",
                    hovertemplate=f"Perimetro Refinado CAD<br>Area = {active_area:.1f} mm²<extra></extra>"
                )
            )
            cent_x_mm = float(cx_mm[:-1].mean())
            cent_y_mm = float(cy_mm[:-1].mean())
            fig.add_trace(
                go.Scatter(
                    x=[cent_x_mm], y=[cent_y_mm],
                    mode="markers",
                    marker=dict(color="#10B981", size=8, symbol="circle"),
                    name="Centroide IA",
                    hovertemplate=f"Centroide IA<br>X = {cent_x_mm:.1f} mm, Y = {cent_y_mm:.1f} mm<extra></extra>"
                )
            )
            fig.add_annotation(
                x=min(cx_mm), y=max(cy_mm) + 6,
                text=f"A (IA) = {active_area:.1f} mm²",
                showarrow=False,
                font=dict(size=12, color="#059669", family="Inter"),
                bgcolor="rgba(255, 255, 255, 0.85)",
                bordercolor="#059669",
                borderwidth=1,
                borderpad=3
            )

    # 3. Modo Auditoria Dual: Clinico vs IA (Ambos contornos para contrastar discrepancias)
    elif view_mode == "Auditoria Dual: Clinico vs IA":
        if contour_pts and len(contour_pts) >= 4:
            gt_x, gt_y = to_mm_poly(contour_pts)
            fig.add_trace(
                go.Scatter(
                    x=gt_x, y=gt_y,
                    mode="lines",
                    line=dict(color="#2563EB", width=2.4),
                    name="Mascara Radiologo (GT)",
                    hovertemplate=f"Mascara Radiologo (GT)<br>Area = {area_mm2:.1f} mm²<extra></extra>"
                )
            )
        if cv_contour_pts and len(cv_contour_pts) >= 4:
            cv_x, cv_y = to_mm_poly(cv_contour_pts)
            fig.add_trace(
                go.Scatter(
                    x=cv_x, y=cv_y,
                    mode="lines",
                    line=dict(color="#059669", width=2.4, dash="dash"),
                    name="Vision Artificial (IA)",
                    hovertemplate=f"Vision Artificial (IA Refinada)<br>Area = {cv_area_mm2:.1f} mm²<br>DICE = {dice_score*100:.1f}%<extra></extra>"
                )
            )
            fig.add_annotation(
                x=min(cv_x), y=max(cv_y) + 6,
                text=f"GT: {area_mm2:.1f} | IA: {cv_area_mm2:.1f} mm² (DICE {dice_score*100:.1f}%)",
                showarrow=False,
                font=dict(size=11, color="#0F172A", family="Inter"),
                bgcolor="rgba(255, 255, 255, 0.90)",
                bordercolor="#94A3B8",
                borderwidth=1,
                borderpad=3
            )

    # Marcadores anatomicos cardinales de orientacion clinica exacta
    fig.add_annotation(x=0, y=half_h - 7, text="ANTERIOR (A) - CARA / FRENTE", showarrow=False, font=dict(color="#2563EB", size=10, family="Inter", weight="bold"))
    fig.add_annotation(x=0, y=-half_h + 7, text="POSTERIOR (P) - OCCIPUCIO", showarrow=False, font=dict(color="#2563EB", size=10, family="Inter", weight="bold"))
    fig.add_annotation(x=-half_w + 12, y=0, text="DERECHA (R)", showarrow=False, font=dict(color="#2563EB", size=10, family="Inter", weight="bold"))
    fig.add_annotation(x=half_w - 12, y=0, text="IZQUIERDA (L)", showarrow=False, font=dict(color="#2563EB", size=10, family="Inter", weight="bold"))

    fig.update_layout(
        title=dict(
            text=f"Corte {slice_idx + 1} de {total_slices} | z = {active_z:.1f} mm | Area: {area_mm2:.1f} mm²",
            font=dict(size=12, family="Inter", color=THEME_MUTED)
        ),
        xaxis=dict(
            title="Eje Sagital X (mm)",
            range=[-half_w - 6, half_w + 6],
            autorange=False,
            gridcolor=THEME_GRID,
            zeroline=False,
            color=THEME_MUTED
        ),
        yaxis=dict(
            title="Eje Coronal Y (mm)",
            range=[-half_h - 6, half_h + 6],
            autorange=False,
            scaleanchor="x",
            scaleratio=1,
            gridcolor=THEME_GRID,
            zeroline=False,
            color=THEME_MUTED
        ),
        margin=dict(l=40, r=20, t=40, b=40),
        paper_bgcolor=THEME_BG,
        plot_bgcolor=THEME_BG,
        font=dict(family="Inter, sans-serif", color=THEME_TEXT)
    )

    return fig


def create_2d_area_profile(
    z_coords: np.ndarray, areas: np.ndarray, method_name: str = "Simpson 1/3"
) -> go.Figure:
    """
    Genera el perfil bidimensional de area transversal A(z) mostrando los puntos
    de telemetria y la curva de interpolacion para cuadratura.
    """
    fig = go.Figure()

    # Curva continua generada con Splines
    if len(z_coords) >= 3:
        spline = NaturalCubicSpline(z_coords, areas)
        z_dense = np.linspace(z_coords[0], z_coords[-1], 150)
        a_dense = spline.evaluate(z_dense)

        # Relleno del area bajo la curva (Integral definida)
        fig.add_trace(
            go.Scatter(
                x=z_dense,
                y=a_dense,
                mode="lines",
                line=dict(color=THEME_BLUE, width=2.5),
                fill="tozeroy",
                fillcolor="rgba(37, 99, 235, 0.10)",
                name="Interpolacion A(z) [Volumen = Integral]"
            )
        )

    # Nodos discretos de telemetria
    fig.add_trace(
        go.Scatter(
            x=z_coords,
            y=areas,
            mode="markers+text",
            marker=dict(color="#101828", size=8, line=dict(color=THEME_BLUE, width=2)),
            name="Cortes Axiales (Telemetria)",
            text=[f"A_{i}" for i in range(len(z_coords))],
            textposition="top center",
            textfont=dict(family="JetBrains Mono", size=10, color=THEME_MUTED),
            hovertemplate="z = %{x:.2f} mm<br>Area = %{y:.2f} mm²<extra></extra>"
        )
    )

    fig.update_layout(
        title=dict(
            text="PERFIL DE AREA SECCIONAL A(z) Y REGION DE INTEGRACION",
            font=dict(size=12, family="Inter", color=THEME_MUTED)
        ),
        xaxis=dict(
            title="Posicion Axial Z (mm)",
            gridcolor=THEME_GRID,
            zerolinecolor=THEME_GRID,
            color=THEME_MUTED
        ),
        yaxis=dict(
            title="Area Transversal A(z) [mm²]",
            gridcolor=THEME_GRID,
            zerolinecolor=THEME_GRID,
            color=THEME_MUTED
        ),
        paper_bgcolor=THEME_PANEL,
        plot_bgcolor=THEME_PANEL,
        margin=dict(l=40, r=20, t=40, b=40),
        font=dict(family="Inter, sans-serif", color=THEME_TEXT),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10, color=THEME_TEXT)
        )
    )
    return fig


def create_error_comparison_chart(benchmark: BenchmarkComparison) -> go.Figure:
    """
    Grafico de barras comparativo de error relativo porcentual entre metodos numericos.
    """
    methods = [
        "Simpson 1/3 Compuesto",
        "Regla del Trapecio",
        "Conteo Voxeles (Riemann)"
    ]
    errors = [
        benchmark.simpson_relative_error_pct,
        benchmark.trapezoid_relative_error_pct,
        benchmark.riemann_relative_error_pct
    ]
    orders = ["O(h^4)", "O(h^2)", "O(h)"]
    colors = [THEME_BLUE, THEME_AMBER, THEME_CRIMSON]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=methods,
            y=errors,
            text=[f"{e:.4f}% ({o})" for e, o in zip(errors, orders)],
            textposition="auto",
            marker=dict(color=colors, line=dict(color=THEME_GRID, width=1)),
            hovertemplate="Metodo: %{x}<br>Error Relativo: %{y:.6f}%<extra></extra>"
        )
    )

    fig.update_layout(
        title=dict(
            text="COMPARATIVA DE ERROR RELATIVO PORCENTUAL FRENTE A SOLUCION EXACTA",
            font=dict(size=12, family="Inter", color=THEME_MUTED)
        ),
        xaxis=dict(gridcolor=THEME_GRID, color=THEME_MUTED),
        yaxis=dict(
            title="Error Relativo (%)",
            gridcolor=THEME_GRID,
            type="log" if max(errors) > 0 and min(errors) > 0 and (max(errors) / max(1e-6, min(errors)) > 20) else "linear",
            color=THEME_MUTED
        ),
        paper_bgcolor=THEME_PANEL,
        plot_bgcolor=THEME_PANEL,
        margin=dict(l=40, r=20, t=40, b=40),
        font=dict(family="Inter, sans-serif", color=THEME_TEXT)
    )
    return fig


# Alias de compatibilidad hacia versiones previas del visualizador
create_3d_lesion_mesh = create_3d_full_brain_visualization

