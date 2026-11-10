"""
Modulo de Evaluacion de Error y Benchmarking Numerico

Calcula cotas teoricas de error de truncamiento O(h^4) y genera
comparativas frente a modelos analiticos exactos (Fantomas esfericos y elipsoidales).
"""

from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np
from src.engine.simpson_engine import NumericalEngine


@dataclass
class BenchmarkComparison:
    """
    Contenedor de resultados comparativos entre metodos para un mismo conjunto de datos.
    """
    slice_thickness_h: float
    num_slices: int
    exact_volume_cm3: float
    simpson_volume_cm3: float
    trapezoid_volume_cm3: float
    riemann_volume_cm3: float
    simpson_relative_error_pct: float
    trapezoid_relative_error_pct: float
    riemann_relative_error_pct: float
    theoretical_error_bound_cm3: float


class ErrorEvaluator:
    """
    Evaluador formal de precision numerica y convergencia asintotica.
    """

    @staticmethod
    def generate_analytical_sphere_profile(
        radius_mm: float, slice_thickness_h: float
    ) -> Dict[str, Any]:
        """
        Genera un perfil analitico exacto de un fantoma esferico de radio R.
        Ecuacion de la esfera: x^2 + y^2 + z^2 <= R^2
        Area transversal a la altura z: A(z) = pi * (R^2 - z^2) para z en [-R, R].
        Volumen exacto: V_exact = (4/3) * pi * R^3.
        
        Derivadas analiticas del area:
        A'(z) = -2 * pi * z
        A''(z) = -2 * pi
        A'''(z) = 0
        A''''(z) = 0
        Nota: Al ser A(z) un polinomio cuadratico en z, la cuarta derivada es identicamente cero.
        Por tanto, la regla de Simpson 1/3 es exacta en este perfil salvo por errores de redondeo.
        """
        n_intervals = max(2, int(round(2.0 * radius_mm / slice_thickness_h)))
        z_coords = np.linspace(-radius_mm, radius_mm, n_intervals + 1)
        effective_h = float(z_coords[1] - z_coords[0])
        
        # Asegurar simetria en los extremos
        areas = np.maximum(0.0, np.pi * (radius_mm**2 - z_coords**2))
        exact_volume_mm3 = (4.0 / 3.0) * np.pi * (radius_mm**3)

        return {
            "name": f"Fantoma Esferico (R = {radius_mm:.1f} mm)",
            "z_coords": z_coords,
            "areas_mm2": areas,
            "exact_volume_mm3": exact_volume_mm3,
            "exact_volume_cm3": exact_volume_mm3 / 1000.0,
            "slice_thickness_h": effective_h,
            "analytical_formula": "V = (4/3) * pi * R^3"
        }

    @staticmethod
    def generate_analytical_ellipsoid_profile(
        semi_a_mm: float, semi_b_mm: float, semi_c_mm: float, slice_thickness_h: float
    ) -> Dict[str, Any]:
        """
        Genera el perfil analitico de un quiste cerebral modelado como un elipsoide triaxial.
        Ecuacion: (x/a)^2 + (y/b)^2 + (z/c)^2 <= 1.
        Area transversal a la altura z: A(z) = pi * a * b * (1 - (z/c)^2) para z en [-c, c].
        Volumen exacto: V = (4/3) * pi * a * b * c.
        """
        # Se garantiza particion simetrica en [-c, c] con paso efectivo h
        n_intervals = max(2, int(round(2.0 * semi_c_mm / slice_thickness_h)))
        z_coords = np.linspace(-semi_c_mm, semi_c_mm, n_intervals + 1)
        effective_h = float(z_coords[1] - z_coords[0])

        factor = np.maximum(0.0, 1.0 - (z_coords / semi_c_mm)**2)
        areas = np.pi * semi_a_mm * semi_b_mm * factor
        exact_volume_mm3 = (4.0 / 3.0) * np.pi * semi_a_mm * semi_b_mm * semi_c_mm

        return {
            "name": f"Quiste Elipsoidal (a={semi_a_mm:.1f}, b={semi_b_mm:.1f}, c={semi_c_mm:.1f} mm)",
            "z_coords": z_coords,
            "areas_mm2": areas,
            "exact_volume_mm3": exact_volume_mm3,
            "exact_volume_cm3": exact_volume_mm3 / 1000.0,
            "slice_thickness_h": effective_h,
            "analytical_formula": "V = (4/3) * pi * a * b * c"
        }

    @classmethod
    def run_comparative_benchmark(
        cls, z_coords: np.ndarray, areas: np.ndarray, exact_volume_cm3: float
    ) -> BenchmarkComparison:
        """
        Ejecuta los tres metodos numericos sobre el mismo conjunto de datos y calcula
        las metricas de error relativo frente a la solucion exacta.
        """
        res_simpson = NumericalEngine.calculate_simpson_13_composite(z_coords, areas)
        res_trapezoid = NumericalEngine.calculate_trapezoidal(z_coords, areas)
        res_riemann = NumericalEngine.calculate_voxel_counting_riemann(z_coords, areas)

        h = res_simpson.slice_thickness_h
        
        # Errores porcentuales relativos
        err_simpson = abs(res_simpson.volume_cm3 - exact_volume_cm3) / exact_volume_cm3 * 100.0
        err_trapezoid = abs(res_trapezoid.volume_cm3 - exact_volume_cm3) / exact_volume_cm3 * 100.0
        err_riemann = abs(res_riemann.volume_cm3 - exact_volume_cm3) / exact_volume_cm3 * 100.0

        # Estimacion de cota teorica de error para Simpson:
        # E_s = -((b - a) / 180) * h^4 * f^(4)(xi)
        # Para polinomios de grado <= 3, f^(4) = 0. En funciones biologicas generales
        # se aproxima la cuarta derivada por diferencias finitas centrales.
        z_range = float(z_coords[-1] - z_coords[0])
        d4_approx = cls._approximate_fourth_derivative(areas, h)
        theoretical_bound_mm3 = ((z_range) / 180.0) * (h**4) * d4_approx
        theoretical_bound_cm3 = theoretical_bound_mm3 / 1000.0

        return BenchmarkComparison(
            slice_thickness_h=h,
            num_slices=len(z_coords),
            exact_volume_cm3=exact_volume_cm3,
            simpson_volume_cm3=res_simpson.volume_cm3,
            trapezoid_volume_cm3=res_trapezoid.volume_cm3,
            riemann_volume_cm3=res_riemann.volume_cm3,
            simpson_relative_error_pct=err_simpson,
            trapezoid_relative_error_pct=err_trapezoid,
            riemann_relative_error_pct=err_riemann,
            theoretical_error_bound_cm3=theoretical_bound_cm3
        )

    @staticmethod
    def _approximate_fourth_derivative(y: np.ndarray, h: float) -> float:
        """
        Calcula una aproximacion discreta de la cota superior del maximo valor
        absoluto de la cuarta derivada mediante diferencias finitas centrales:
        f^(4)(z) approx (f(z+2h) - 4f(z+h) + 6f(z) - 4f(z-h) + f(z-2h)) / h^4.
        """
        if len(y) < 5 or h <= 0:
            return 0.0
        
        d4_vals = []
        for i in range(2, len(y) - 2):
            val = (y[i+2] - 4.0*y[i+1] + 6.0*y[i] - 4.0*y[i-1] + y[i-2]) / (h**4)
            d4_vals.append(abs(val))
            
        return float(np.max(d4_vals)) if d4_vals else 0.0
