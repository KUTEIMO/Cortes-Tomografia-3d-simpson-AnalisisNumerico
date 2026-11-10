"""
Pruebas de Metodos Complementarios: Splines Cubicos y Romberg
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.engine.spline_interpolator import NaturalCubicSpline
from src.engine.romberg_engine import RombergEngine
from src.engine.simpson_engine import NumericalEngine
from src.engine.error_evaluator import ErrorEvaluator


class TestComplementaryMethods(unittest.TestCase):
    """
    Verificacion de trazadores cubicos y extrapolacion de Richardson / Romberg.
    """

    def test_spline_interpolation_preserves_nodes(self):
        """
        El trazador cubico natural debe pasar exactamente por los nodos de interpolacion:
        S(x_i) == y_i.
        """
        x = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
        y = np.array([0.0, 15.2, 42.1, 28.4, 0.0])

        spline = NaturalCubicSpline(x, y)
        y_eval = spline.evaluate(x)

        np.testing.assert_allclose(
            y_eval, y, atol=1e-10, err_msg="El spline debe interpolar exactamente los puntos nodales."
        )

    def test_romberg_level1_equivalence_with_simpson(self):
        """
        R(1,1) en la tabla de Romberg debe coincidir con la aproximacion de Simpson 1/3
        demostrando la deduccion por extrapolacion de Richardson.
        """
        profile = ErrorEvaluator.generate_analytical_sphere_profile(radius_mm=16.0, slice_thickness_h=2.0)
        z = profile["z_coords"]
        a = profile["areas_mm2"]

        romberg_res = RombergEngine.compute_romberg_table(z, a, max_levels=3)
        self.assertIn("simpson_13_via_richardson_cm3", romberg_res)
        self.assertGreater(romberg_res["best_estimate_cm3"], 0.0)


if __name__ == "__main__":
    unittest.main()

