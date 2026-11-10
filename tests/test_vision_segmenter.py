"""
Pruebas Unitarias del Modulo de Vision Artificial y Segmentacion
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vision.segmenter import BrainLesionSegmenter


class TestVisionSegmenter(unittest.TestCase):
    """
    Verificacion de procesamiento de imagenes, extraccion de contornos y calculo de area.
    """

    def test_segment_case_01_consistency(self):
        case_dir = os.path.join("data", "kaggle_cases", "PATIENT_01_TCGA_4944")
        packets, images, info = BrainLesionSegmenter.process_case_series(case_dir)

        self.assertEqual(len(packets), 20)
        self.assertEqual(len(images), 20)
        self.assertEqual(info["patient_id"], "PATIENT_01_TCGA_4944")
        
        # Verificar que el corte pico este en el indice 9
        areas = [p.cross_sectional_area_mm2 for p in packets]
        max_idx = areas.index(max(areas))
        self.assertEqual(max_idx, 9, "El corte de mayor extension lesional debe ser el indice 9.")
        self.assertGreater(areas[9], 5000.0, "El corte pico debe superar los 5000 mm2.")


if __name__ == "__main__":
    unittest.main()
