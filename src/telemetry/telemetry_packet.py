"""
Modulo de Definicion del Paquete de Telemetria Radiologica DICOM

Define la estructura de datos del paquete de telemetria clinica (Telemetric Packet)
transmitido desde la consola del tomografo o servidor PACS hacia la estacion de diagnostico.
Incluye validacion de integridad mediante CRC32.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple
import zlib
import json


@dataclass
class TelemetryPacket:
    """
    Estructura estandar de un paquete telemétrico de corte tomografico axial.
    """
    # 1. Cabecera y Control de Sesion
    packet_seq_num: int
    modality: str  # e.g., 'CT' (Tomografia Computarizada) o 'MR' (Resonancia Magnetica)
    patient_anon_id: str
    timestamp_utc: str

    # 2. Telemetria Espacial (Geometrica)
    slice_index: int
    z_position_mm: float
    slice_thickness_h_mm: float
    pixel_spacing_x_mm: float
    pixel_spacing_y_mm: float
    matrix_dim_x: int
    matrix_dim_y: int

    # 3. Telemetria de Adquisicion Fisica (Dosimetria y Hardware)
    tube_voltage_kvp: float
    tube_current_ma: float
    gantry_tilt_deg: float
    rescale_slope: float
    rescale_intercept: float

    # 4. Telemetria Lesional (Segmentacion Seccional de la Patologia)
    roi_pixel_count: int
    cross_sectional_area_mm2: float
    equivalent_diameter_mm: float
    centroid_x_mm: float
    centroid_y_mm: float
    hounsfield_mean_hu: float

    # 5. Integridad de Transmision
    crc32_checksum: int = 0

    def compute_crc32(self) -> int:
        """
        Calcula la suma de verificacion ciclica CRC32 del paquete
        sobre sus campos esenciales para certificar la integridad clinica.
        """
        payload = (
            f"{self.packet_seq_num}:{self.modality}:{self.slice_index}:"
            f"{self.z_position_mm:.4f}:{self.slice_thickness_h_mm:.4f}:"
            f"{self.cross_sectional_area_mm2:.4f}:{self.roi_pixel_count}"
        )
        return zlib.crc32(payload.encode("utf-8"))

    def stamp_crc32(self):
        """Asigna el CRC32 calculado al paquete."""
        self.crc32_checksum = self.compute_crc32()

    def verify_integrity(self) -> bool:
        """Verifica que el paquete no haya sufrido corrupcion durante la transmision."""
        return self.crc32_checksum == self.compute_crc32()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TelemetryPacket":
        pkt = cls(**data)
        return pkt

