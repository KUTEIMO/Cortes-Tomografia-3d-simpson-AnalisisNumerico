"""
Pruebas Unitarias del Modulo de Telemetria DICOM
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.telemetry.telemetry_packet import TelemetryPacket
from src.telemetry.stream_simulator import TelemetryStreamSimulator


class TestTelemetryStream(unittest.TestCase):
    """
    Verificacion de empaquetado, CRC32 e ingestion de telemetria tomografica.
    """

    def test_crc32_integrity_check(self):
        """
        Verifica que el checksum CRC32 detecte alteraciones en la carga util del paquete.
        """
        pkt = TelemetryPacket(
            packet_seq_num=1,
            modality="CT",
            patient_anon_id="ANON-TEST-001",
            timestamp_utc="2026-11-04T12:00:00Z",
            slice_index=0,
            z_position_mm=10.0,
            slice_thickness_h_mm=2.5,
            pixel_spacing_x_mm=0.5,
            pixel_spacing_y_mm=0.5,
            matrix_dim_x=512,
            matrix_dim_y=512,
            tube_voltage_kvp=120.0,
            tube_current_ma=200.0,
            gantry_tilt_deg=0.0,
            rescale_slope=1.0,
            rescale_intercept=-1024.0,
            roi_pixel_count=1200,
            cross_sectional_area_mm2=300.0,
            equivalent_diameter_mm=19.54,
            centroid_x_mm=128.0,
            centroid_y_mm=128.0,
            hounsfield_mean_hu=10.2
        )
        pkt.stamp_crc32()
        self.assertTrue(pkt.verify_integrity(), "El paquete recien sellado debe ser integro.")

        # Simular corrupcion telemétrica en el valor del area
        pkt.cross_sectional_area_mm2 += 5.0
        self.assertFalse(pkt.verify_integrity(), "El checksum CRC32 debe detectar la alteracion telemétrica.")

    def test_synthetic_stream_generation_and_extraction(self):
        """
        Verifica la generacion de la serie sintetica y la extraccion correcta de vectores numericos.
        """
        sim = TelemetryStreamSimulator()
        packets = sim.generate_synthetic_cyst_stream(num_slices=11, slice_thickness_h=2.5)

        self.assertEqual(len(packets), 11)
        z, a, h = sim.extract_numerical_arrays()

        self.assertEqual(len(z), 11)
        self.assertEqual(len(a), 11)
        self.assertAlmostEqual(h, 2.5, places=3)
        self.assertTrue(all(p.verify_integrity() for p in packets))


if __name__ == "__main__":
    unittest.main()

