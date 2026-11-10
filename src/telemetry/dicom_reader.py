"""
Modulo de Lectura y Extraccion de Cabeceras DICOM Estandar

Extrae etiquetas espaciales y dosimetricas del estandar internacional DICOM (.dcm):
- (0028, 0030) Pixel Spacing (Δx, Δy)
- (0018, 0050) Slice Thickness (h)
- (0020, 0032) Image Position Patient (Z)
- (0028, 1052) Rescale Intercept
- (0028, 1053) Rescale Slope
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from src.telemetry.telemetry_packet import TelemetryPacket

# Importacion condicional y segura de pydicom
try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


class DicomHeaderReader:
    """
    Lector de metadatos de archivos DICOM estandar para la extraccion de telemetria.
    """

    @staticmethod
    def is_pydicom_available() -> bool:
        return HAS_PYDICOM

    @classmethod
    def parse_dicom_file(cls, file_path: str, slice_index: int = 0) -> Optional[TelemetryPacket]:
        """
        Lee un archivo DICOM binario y extrae el paquete de telemetria.
        """
        if not HAS_PYDICOM:
            return None

        path = Path(file_path)
        if not path.exists():
            return None

        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=False)

            # Extraccion de etiquetas estandar DICOM
            modality = getattr(ds, "Modality", "CT")
            anon_id = getattr(ds, "PatientID", "ANON-PATIENT")
            thickness = float(getattr(ds, "SliceThickness", 2.5))

            # Image Position Patient (x, y, z)
            img_pos = getattr(ds, "ImagePositionPatient", [0.0, 0.0, float(slice_index * thickness)])
            z_pos = float(img_pos[2])

            # Pixel Spacing [row_spacing, col_spacing]
            pixel_spacing = getattr(ds, "PixelSpacing", [0.5, 0.5])
            ps_x = float(pixel_spacing[0])
            ps_y = float(pixel_spacing[1])

            # Extraccion o estimacion de area desde PixelData si existe mascara/umbral
            area_mm2 = 0.0
            pixel_count = 0
            mean_hu = 0.0

            if hasattr(ds, "pixel_array"):
                arr = ds.pixel_array
                slope = float(getattr(ds, "RescaleSlope", 1.0))
                intercept = float(getattr(ds, "RescaleIntercept", 0.0))
                hu_arr = arr.astype(np.float64) * slope + intercept

                # Segmentacion por rango de Unidades Hounsfield tipico de quistes (liquido: 0 a 20 HU)
                cyst_mask = (hu_arr >= 0.0) & (hu_arr <= 25.0)
                pixel_count = int(np.sum(cyst_mask))
                area_mm2 = pixel_count * (ps_x * ps_y)
                if pixel_count > 0:
                    mean_hu = float(np.mean(hu_arr[cyst_mask]))

            pkt = TelemetryPacket(
                packet_seq_num=slice_index + 1,
                modality=modality,
                patient_anon_id=anon_id,
                timestamp_utc="2026-11-04T12:00:00Z",
                slice_index=slice_index,
                z_position_mm=z_pos,
                slice_thickness_h_mm=thickness,
                pixel_spacing_x_mm=ps_x,
                pixel_spacing_y_mm=ps_y,
                matrix_dim_x=int(getattr(ds, "Columns", 512)),
                matrix_dim_y=int(getattr(ds, "Rows", 512)),
                tube_voltage_kvp=float(getattr(ds, "KVP", 120.0)),
                tube_current_ma=float(getattr(ds, "XRayTubeCurrent", 200.0)),
                gantry_tilt_deg=float(getattr(ds, "GantryDetectorTilt", 0.0)),
                rescale_slope=float(getattr(ds, "RescaleSlope", 1.0)),
                rescale_intercept=float(getattr(ds, "RescaleIntercept", 0.0)),
                roi_pixel_count=pixel_count,
                cross_sectional_area_mm2=area_mm2,
                equivalent_diameter_mm=float(2.0 * np.sqrt(area_mm2 / np.pi)) if area_mm2 > 0 else 0.0,
                centroid_x_mm=256.0 * ps_x,
                centroid_y_mm=256.0 * ps_y,
                hounsfield_mean_hu=mean_hu
            )
            pkt.stamp_crc32()
            return pkt
        except Exception:
            return None

