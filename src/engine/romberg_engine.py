"""
Modulo de Integracion de Romberg y Extrapolacion de Richardson

Metodo complementario de Analisis Numerico:
Demuestra formalmente que la Regla Compuesta de Simpson 1/3 (O(h^4)) es el resultado
directo de aplicar la primera iteracion de la Extrapolacion de Richardson sobre la
Regla del Trapecio:
R(1,1) = (4 * R(1,0) - R(0,0)) / 3 == Simpson 1/3
"""

from typing import Tuple, List, Dict, Any
import numpy as np


class RombergEngine:
    """
    Motor de extrapolacion de Richardson y construccion de la tabla triangular de Romberg.
    """

    @classmethod
    def compute_romberg_table(
        cls, z_coords: np.ndarray, areas: np.ndarray, max_levels: int = 3
    ) -> Dict[str, Any]:
        """
        Construye la tabla de Romberg a partir de un conjunto de datos equiparticionados
        o densificados mediante splines.
        
        R[j, 0] = Trapecio con paso h / 2^j
        R[j, k] = (4^k * R[j, k-1] - R[j-1, k-1]) / (4^k - 1)
        """
        z_arr = np.asarray(z_coords, dtype=np.float64)
        a_arr = np.asarray(areas, dtype=np.float64)

        n_points = len(z_arr)
        # Para Romberg clasico se requiere que el numero de intervalos sea potencia de 2: 2^m
        # Si no lo es, se utiliza submuestreo de niveles disponibles
        levels = min(max_levels, int(np.log2(max(2, n_points - 1))) + 1)
        levels = max(2, levels)

        table = np.zeros((levels, levels), dtype=np.float64)

        # Nivel 0: Trapecio con el paso mas grueso disponible
        h_base = float(z_arr[-1] - z_arr[0])
        # R[0, 0] = (b - a)/2 * (f(a) + f(b))
        table[0, 0] = 0.5 * h_base * (a_arr[0] + a_arr[-1])

        # Niveles sucesivos R[j, 0]
        for j in range(1, levels):
            n_sub = 2**j
            step = (len(z_arr) - 1) // n_sub
            if step < 1:
                step = 1
            
            sub_indices = np.linspace(0, len(z_arr) - 1, n_sub + 1, dtype=int)
            sub_z = z_arr[sub_indices]
            sub_a = a_arr[sub_indices]
            
            diffs = np.diff(sub_z)
            h_eff = float(diffs[0]) if len(diffs) > 0 else 1.0
            table[j, 0] = (h_eff / 2.0) * (sub_a[0] + 2.0 * np.sum(sub_a[1:-1]) + sub_a[-1])

        # Extrapolacion de Richardson para columnas k >= 1
        for k in range(1, levels):
            factor = 4.0**k
            for j in range(k, levels):
                table[j, k] = (factor * table[j, k - 1] - table[j - 1, k - 1]) / (factor - 1.0)

        # R[1, 1] corresponde exactamente a Simpson 1/3
        simpson_from_romberg = table[1, 1]

        return {
            "romberg_table_mm3": table,
            "levels": levels,
            "simpson_13_via_richardson_mm3": float(simpson_from_romberg),
            "simpson_13_via_richardson_cm3": float(simpson_from_romberg) / 1000.0,
            "best_estimate_cm3": float(table[-1, -1]) / 1000.0,
            "error_orders_by_column": [f"O(h^{2*(k+1)})" for k in range(levels)]
        }

