"""
Modulo del Motor de Integracion Numerica para NeuroCalc

Implementacion rigurosa de metodos de cuadratura de Newton-Cotes:
- Regla Compuesta de Simpson 1/3 (Orden de error O(h^4))
- Regla de Simpson 3/8 para correccion de paridad en intervalos impares
- Regla Compuesta del Trapecio (Orden O(h^2))
- Conteo de Voxeles / Suma de Riemann (Orden O(h))
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np


@dataclass
class IntegrationResult:
    """
    Estructura de datos que encapsula los resultados del calculo volumetrico.
    """
    volume_mm3: float
    volume_cm3: float
    method_used: str
    num_slices: int
    num_intervals: int
    slice_thickness_h: float
    is_equidistant: bool
    intermediate_steps: Dict[str, Any]
    error_order: str


class NumericalEngine:
    """
    Motor matematico principal encargado de la resolucion de integrales
    definidas para el calculo del volumen V = integral_{z0}^{zn} A(z) dz.
    """

    TOLERANCE_EQUIDISTANT = 1e-4

    @staticmethod
    def verify_equidistance(z_coords: np.ndarray) -> Tuple[bool, float]:
        """
        Verifica que los cortes axiales esten distribuidos uniformemente (h constante).
        Retorna (es_equidistante, h_promedio).
        """
        if len(z_coords) < 2:
            return False, 0.0
        
        diffs = np.diff(z_coords)
        h_mean = float(np.mean(diffs))
        is_equidistant = bool(np.all(np.abs(diffs - h_mean) < NumericalEngine.TOLERANCE_EQUIDISTANT))
        return is_equidistant, h_mean

    @classmethod
    def calculate_simpson_13_composite(
        cls, z_coords: np.ndarray, areas: np.ndarray
    ) -> IntegrationResult:
        """
        Calcula el volumen empleando la Regla Compuesta de Simpson 1/3 con
        adaptacion hibrida de Simpson 3/8 para conjuntos con numero impar de intervalos.
        
        Parametros:
            z_coords: Coordenadas axiales en mm (ordenadas ascendentemente).
            areas: Areas seccionales transversales A(z_i) en mm^2.
            
        Retorna:
            IntegrationResult con los valores calculados y trazabilidad paso a paso.
        """
        z_arr = np.asarray(z_coords, dtype=np.float64)
        a_arr = np.asarray(areas, dtype=np.float64)

        n_slices = len(z_arr)
        if n_slices != len(a_arr):
            raise ValueError("Las dimensiones de coordenadas Z y areas deben ser identicas.")
        
        if n_slices < 3:
            # Fallback al Trapecio si hay menos de 3 puntos (no alcanza para Simpson)
            return cls.calculate_trapezoidal(z_arr, a_arr)

        is_equidistant, h = cls.verify_equidistance(z_arr)
        if not is_equidistant:
            # Advertencia: espaciado variable. Se emplea suma trapezoidal generalizada
            return cls.calculate_trapezoidal_non_uniform(z_arr, a_arr)

        n_intervals = n_slices - 1

        # Caso 1: Intervalos pares (n % 2 == 0) -> Simpson 1/3 Compuesto Puro
        if n_intervals % 2 == 0:
            sum_odd_idx = np.sum(a_arr[1:n_intervals:2])   # Terminos con coeficiente 4
            sum_even_idx = np.sum(a_arr[2:n_intervals:2])  # Terminos con coeficiente 2
            
            integral_val = (h / 3.0) * (a_arr[0] + 4.0 * sum_odd_idx + 2.0 * sum_even_idx + a_arr[-1])
            method_desc = "Simpson 1/3 Compuesto Puro"
            error_order = "O(h^4)"
            
            steps = {
                "h": h,
                "A_0": float(a_arr[0]),
                "A_n": float(a_arr[-1]),
                "sum_impares_coef4": float(sum_odd_idx),
                "sum_pares_coef2": float(sum_even_idx),
                "factor_h_tercios": h / 3.0,
                "paridad_intervalos": "Par (Valido para 1/3 estricto)"
            }

        # Caso 2: Intervalos impares (n % 2 != 0) -> Hibrido Simpson 1/3 + Simpson 3/8
        else:
            if n_intervals < 3:
                # Caso extremo n = 1 -> no es posible Simpson 3/8
                return cls.calculate_trapezoidal(z_arr, a_arr)

            # Subconjunto 1: Simpson 1/3 para los primeros (n - 3) intervalos
            cut_idx = n_intervals - 3
            z_s13 = z_arr[: cut_idx + 1]
            a_s13 = a_arr[: cut_idx + 1]
            
            if cut_idx > 0:
                sum_odd = np.sum(a_s13[1:cut_idx:2])
                sum_even = np.sum(a_s13[2:cut_idx:2])
                integral_s13 = (h / 3.0) * (a_s13[0] + 4.0 * sum_odd + 2.0 * sum_even + a_s13[-1])
            else:
                integral_s13 = 0.0

            # Subconjunto 2: Simpson 3/8 para los ultimos 3 intervalos (4 puntos)
            a_s38 = a_arr[cut_idx:]
            integral_s38 = (3.0 * h / 8.0) * (a_s38[0] + 3.0 * a_s38[1] + 3.0 * a_s38[2] + a_s38[3])

            integral_val = integral_s13 + integral_s38
            method_desc = "Hibrido Simpson 1/3 (n-3) + Simpson 3/8 (3 ultimos)"
            error_order = "O(h^4) Global Preservado"

            steps = {
                "h": h,
                "cut_index": cut_idx,
                "integral_parcial_simpson_13": float(integral_s13),
                "integral_parcial_simpson_38": float(integral_s38),
                "paridad_intervalos": "Impar (Corregido con Simpson 3/8 para evitar degradacion a O(h^2))"
            }

        vol_mm3 = float(integral_val)
        vol_cm3 = vol_mm3 / 1000.0  # 1 cm^3 = 1000 mm^3

        return IntegrationResult(
            volume_mm3=vol_mm3,
            volume_cm3=vol_cm3,
            method_used=method_desc,
            num_slices=n_slices,
            num_intervals=n_intervals,
            slice_thickness_h=h,
            is_equidistant=True,
            intermediate_steps=steps,
            error_order=error_order
        )

    @classmethod
    def calculate_trapezoidal(
        cls, z_coords: np.ndarray, areas: np.ndarray
    ) -> IntegrationResult:
        """
        Calcula el volumen empleando la Regla Compuesta del Trapecio (Orden O(h^2)).
        Utilizado como linea base de comparacion.
        """
        z_arr = np.asarray(z_coords, dtype=np.float64)
        a_arr = np.asarray(areas, dtype=np.float64)
        n_slices = len(z_arr)
        n_intervals = n_slices - 1
        is_equidistant, h = cls.verify_equidistance(z_arr)

        if is_equidistant:
            integral_val = (h / 2.0) * (a_arr[0] + 2.0 * np.sum(a_arr[1:-1]) + a_arr[-1])
        else:
            diffs = np.diff(z_arr)
            integral_val = np.sum(0.5 * (a_arr[:-1] + a_arr[1:]) * diffs)
            h = float(np.mean(diffs))

        vol_mm3 = float(integral_val)
        return IntegrationResult(
            volume_mm3=vol_mm3,
            volume_cm3=vol_mm3 / 1000.0,
            method_used="Regla Compuesta del Trapecio",
            num_slices=n_slices,
            num_intervals=n_intervals,
            slice_thickness_h=h,
            is_equidistant=is_equidistant,
            intermediate_steps={"formula": "(h/2) * [A0 + 2*sum(Ai) + An]"},
            error_order="O(h^2)"
        )

    @classmethod
    def calculate_trapezoidal_non_uniform(
        cls, z_coords: np.ndarray, areas: np.ndarray
    ) -> IntegrationResult:
        """
        Calcula la integral trapezoidal sobre cortes con separacion no uniforme.
        """
        diffs = np.diff(z_coords)
        integral_val = np.sum(0.5 * (areas[:-1] + areas[1:]) * diffs)
        vol_mm3 = float(integral_val)
        return IntegrationResult(
            volume_mm3=vol_mm3,
            volume_cm3=vol_mm3 / 1000.0,
            method_used="Trapecio Compuesto Generalizado (No Uniforme)",
            num_slices=len(z_coords),
            num_intervals=len(z_coords) - 1,
            slice_thickness_h=float(np.mean(diffs)),
            is_equidistant=False,
            intermediate_steps={"advertencia": "Espaciado axial no uniforme detectado en telemetria"},
            error_order="O(h_max^2)"
        )

    @classmethod
    def calculate_voxel_counting_riemann(
        cls, z_coords: np.ndarray, areas: np.ndarray
    ) -> IntegrationResult:
        """
        Calcula el volumen empleando Conteo Directo de Voxeles (Suma de Riemann de orden O(h)).
        Representa el metodo estandar rapido de visores convencionales.
        """
        is_equidistant, h = cls.verify_equidistance(z_coords)
        if not is_equidistant:
            diffs = np.diff(z_coords)
            h = float(np.mean(diffs))
        
        integral_val = np.sum(areas) * h
        vol_mm3 = float(integral_val)
        return IntegrationResult(
            volume_mm3=vol_mm3,
            volume_cm3=vol_mm3 / 1000.0,
            method_used="Conteo de Voxeles (Suma de Riemann Izquierda/Punto Medio)",
            num_slices=len(z_coords),
            num_intervals=len(z_coords) - 1,
            slice_thickness_h=h,
            is_equidistant=is_equidistant,
            intermediate_steps={"formula": "sum(A_i) * h"},
            error_order="O(h)"
        )

