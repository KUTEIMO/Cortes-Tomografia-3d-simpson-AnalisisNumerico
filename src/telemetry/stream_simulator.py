"""
Modulo de Simulacion de Streaming y Buffer Telemétrico PACS/DICOM

Simula la transmision en tiempo real de paquetes de telemetria radiologica
hacia la estacion de trabajo NeuroCalc.
"""

import time
import json
from pathlib import Path
from typing import List, Generator, Dict, Any, Optional, Tuple
import numpy as np

from src.telemetry.telemetry_packet import TelemetryPacket


class TelemetryStreamSimulator:
    """
    Gestor de ingestion de telemetria tomografica y buffer de procesamiento.
    """

    def __init__(self):
        self.packet_buffer: List[TelemetryPacket] = []
        self.corrupted_packets_count: int = 0

    def load_from_json_file(self, file_path: str) -> List[TelemetryPacket]:
        """
        Carga una serie completa de telemetria almacenada en formato JSON estandar.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Archivo de telemetria no encontrado: {file_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        packets = []
        self.corrupted_packets_count = 0
        raw_list = data if isinstance(data, list) else data.get("slices", [])

        for item in raw_list:
            pkt = TelemetryPacket.from_dict(item)
            if pkt.verify_integrity():
                packets.append(pkt)
            else:
                self.corrupted_packets_count += 1

        # Ordenar estrictamente por coordenada axial Z ascendente
        packets.sort(key=lambda p: p.z_position_mm)
        self.packet_buffer = packets
        return packets

    def generate_synthetic_cyst_stream(
        self,
        num_slices: int = 11,
        slice_thickness_h: float = 2.5,
        semi_a_mm: float = 16.0,
        semi_b_mm: float = 22.0,
        semi_c_mm: float = 12.5,
        center_z_mm: float = 0.0,
        pixel_spacing_mm: float = 0.488281,
        noise_factor: float = 0.0
    ) -> List[TelemetryPacket]:
        """
        Genera una serie sintetica controlada de telemetria con datos morfologicos
        de un quiste aracnoideo intracraneal.
        """
        packets = []
        z_min = center_z_mm - semi_c_mm
        z_max = center_z_mm + semi_c_mm
        z_coords = np.linspace(z_min, z_max, num_slices)

        voxel_area = pixel_spacing_mm * pixel_spacing_mm

        for idx, z in enumerate(z_coords):
            # Perfil elipsoidal de la seccion transversal
            norm_z = (z - center_z_mm) / semi_c_mm
            factor = max(0.0, 1.0 - norm_z**2)
            base_area = np.pi * semi_a_mm * semi_b_mm * factor

            if noise_factor > 0:
                noise = np.random.uniform(-noise_factor, noise_factor) * base_area
                area = max(0.0, base_area + noise)
            else:
                area = base_area

            pixel_count = int(round(area / voxel_area)) if voxel_area > 0 else 0
            equiv_diameter = 2.0 * np.sqrt(area / np.pi) if area > 0 else 0.0

            pkt = TelemetryPacket(
                packet_seq_num=idx + 1,
                modality="CT",
                patient_anon_id="ANON-NEURO-7729",
                timestamp_utc=f"2026-11-04T10:{idx:02d}:00Z",
                slice_index=idx,
                z_position_mm=float(z),
                slice_thickness_h_mm=float(slice_thickness_h),
                pixel_spacing_x_mm=float(pixel_spacing_mm),
                pixel_spacing_y_mm=float(pixel_spacing_mm),
                matrix_dim_x=512,
                matrix_dim_y=512,
                tube_voltage_kvp=120.0,
                tube_current_ma=250.0,
                gantry_tilt_deg=0.0,
                rescale_slope=1.0,
                rescale_intercept=-1024.0,
                roi_pixel_count=pixel_count,
                cross_sectional_area_mm2=float(area),
                equivalent_diameter_mm=float(equiv_diameter),
                centroid_x_mm=256.0 * pixel_spacing_mm,
                centroid_y_mm=256.0 * pixel_spacing_mm,
                hounsfield_mean_hu=8.4  # Tipico de liquido cefalorraquideo (0 a 15 HU)
            )
            pkt.stamp_crc32()
            packets.append(pkt)

        self.packet_buffer = packets
        return packets

    def extract_numerical_arrays(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Extrae los vectores ordenados de coordenadas z_coords, areas y el espesor h
        para su alimentacion directa al motor numerico de Simpson.
        """
        if not self.packet_buffer:
            raise ValueError("El buffer de telemetria se encuentra vacio.")

        z_coords = np.array([p.z_position_mm for p in self.packet_buffer], dtype=np.float64)
        areas = np.array([p.cross_sectional_area_mm2 for p in self.packet_buffer], dtype=np.float64)
        h = float(self.packet_buffer[0].slice_thickness_h_mm)
        return z_coords, areas, h

    def export_to_json(self, file_path: str):
        """Exporta el buffer actual a un archivo JSON de telemetria."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [p.to_dict() for p in self.packet_buffer]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
