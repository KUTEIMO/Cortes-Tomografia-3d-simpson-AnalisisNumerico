"""
Pruebas Unitarias de Precision y Convergencia Matematica

Valida el motor numerico frente a soluciones analiticas exactas.
"""

import os
import sys
import unittest
import numpy as np

# Inyeccion de ruta raiz para portabilidad en cualquier maquina
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.engine.simpson_engine import NumericalEngine
from src.engine.error_evaluator import ErrorEvaluator


class TestSimpsonConvergence(unittest.TestCase):
    """
    Casos de prueba para verificar la exactitud y convergencia del motor de Simpson.
    """

    def setUp(self):
        self.radius = 20.0  # Radio de 20 mm (40 mm de diametro)
        self.slice_thickness = 2.0  # Cortes cada 2 mm

    def test_analytical_sphere_simpson_exactness(self):
        """
        Para A(z) = pi * (R^2 - z^2), la funcion es un polinomio cuadratico.
        La regla de Simpson 1/3 interpola polinomios de segundo y tercer grado con error cero.
        Por tanto, el error relativo teorico debe ser virtualmente nulo (< 0.05%).
        """
        profile = ErrorEvaluator.generate_analytical_sphere_profile(
            radius_mm=self.radius, slice_thickness_h=self.slice_thickness
        )
        res = NumericalEngine.calculate_simpson_13_composite(
            profile["z_coords"], profile["areas_mm2"]
        )

        exact_vol_cm3 = profile["exact_volume_cm3"]
        calc_vol_cm3 = res.volume_cm3

        rel_error_pct = abs(calc_vol_cm3 - exact_vol_cm3) / exact_vol_cm3 * 100.0

        self.assertLess(
            rel_error_pct,
            0.05,
            f"El error relativo de Simpson 1/3 ({rel_error_pct:.4f}%) supera la tolerancia de precision.",
        )
        self.assertEqual(res.error_order, "O(h^4)")

    def test_odd_intervals_hybrid_handling(self):
        """
        Verifica que cuando el numero de intervalos n es impar, el motor active
        el algoritmo hibrido (Simpson 1/3 + Simpson 3/8) y preserve la precision.
        """
        # Seleccionar intencionalmente un corte que produzca intervalos impares
        profile = ErrorEvaluator.generate_analytical_sphere_profile(
            radius_mm=self.radius, slice_thickness_h=self.slice_thickness
        )
        z_coords = profile["z_coords"][:-1]  # Remover un punto para alterar paridad
        areas = profile["areas_mm2"][:-1]
        n_intervals = len(z_coords) - 1

        if n_intervals % 2 == 0:
            z_coords = z_coords[:-1]
            areas = areas[:-1]
            n_intervals = len(z_coords) - 1

        self.assertTrue(n_intervals % 2 != 0, "Se requiere n impar para esta prueba.")

        res = NumericalEngine.calculate_simpson_13_composite(z_coords, areas)

        self.assertIn("Hibrido", res.method_used)
        self.assertEqual(res.error_order, "O(h^4) Global Preservado")
        self.assertIn("integral_parcial_simpson_38", res.intermediate_steps)

    def test_comparative_superiority_over_riemann_and_trapezoid(self):
        """
        Demuestra la superioridad de convergencia de Simpson 1/3 frente a la Suma de Riemann
        (conteo de voxeles clasico) y la regla del Trapecio.
        """
        semi_a, semi_b, semi_c = 15.0, 22.0, 18.0
        h = 2.5
        ellipsoid = ErrorEvaluator.generate_analytical_ellipsoid_profile(
            semi_a, semi_b, semi_c, slice_thickness_h=h
        )

        benchmark = ErrorEvaluator.run_comparative_benchmark(
            ellipsoid["z_coords"], ellipsoid["areas_mm2"], ellipsoid["exact_volume_cm3"]
        )

        # Simpson debe tener un error relativo significativamente menor que Riemann
        self.assertLess(
            benchmark.simpson_relative_error_pct,
            benchmark.riemann_relative_error_pct,
            "Simpson 1/3 debe ser cuantitativamente mas preciso que el conteo de voxeles de Riemann.",
        )
        self.assertLess(
            benchmark.simpson_relative_error_pct,
            benchmark.trapezoid_relative_error_pct,
            "Simpson 1/3 debe superar en precision a la regla del Trapecio.",
        )


if __name__ == "__main__":
    unittest.main()
