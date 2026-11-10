"""
Modulo de Interpolacion por Trazadores Cubicos Naturales (Cubic Splines)

Metodo complementario de Analisis Numerico:
Permite reconstruir el perfil continuo de area seccional A(z) con continuidad C^2
(continuidad de funcion, primera y segunda derivada) a partir de los cortes discretos DICOM.
Esto permite densificar el muestreo para la integracion de Simpson o modelar la morfologia
biologica continua de la lesion cerebral.
"""

from typing import Tuple, List
import numpy as np


class NaturalCubicSpline:
    """
    Implementacion rigurosa de Trazadores Cubicos Naturales mediante la resolucion
    del sistema tridiagonal de momentos (segundas derivadas M_i).
    Condicion de frontera natural: M_0 = 0, M_n = 0.
    """

    def __init__(self, x_coords: np.ndarray, y_values: np.ndarray):
        self.x = np.asarray(x_coords, dtype=np.float64)
        self.y = np.asarray(y_values, dtype=np.float64)
        self.n = len(self.x) - 1

        if self.n < 2:
            raise ValueError("Se requieren al menos 3 puntos para trazar splines cubicos.")

        # Vector de espaciados h_i = x_{i+1} - x_i
        self.h = np.diff(self.x)
        if np.any(self.h <= 0):
            raise ValueError("Las coordenadas x deben estar estrictamente ordenadas en forma creciente.")

        # Coeficientes a_i, b_i, c_i, d_i para cada tramo i en [x_i, x_{i+1}]
        # S_i(x) = a_i + b_i*(x - x_i) + c_i*(x - x_i)^2 + d_i*(x - x_i)^3
        self.a = np.zeros(self.n)
        self.b = np.zeros(self.n)
        self.c = np.zeros(self.n)
        self.d = np.zeros(self.n)

        self._fit()

    def _fit(self):
        """
        Construye y resuelve el sistema tridiagonal de ecuaciones para las segundas derivadas.
        """
        n = self.n
        h = self.h
        y = self.y

        # Construccion del sistema tridiagonal A * M = rhs
        # A tiene dimension (n-1) x (n-1) para los momentos interiores M_1, ..., M_{n-1}
        size = n - 1
        alpha = np.zeros(size)
        beta = np.zeros(size)
        gamma = np.zeros(size)
        rhs = np.zeros(size)

        for i in range(size):
            # En notacion global: k = i + 1
            k = i + 1
            if i > 0:
                alpha[i] = h[k - 1]  # Subdiagonal
            beta[i] = 2.0 * (h[k - 1] + h[k])  # Diagonal principal
            if i < size - 1:
                gamma[i] = h[k]  # Superdiagonal

            term1 = (y[k + 1] - y[k]) / h[k]
            term2 = (y[k] - y[k - 1]) / h[k - 1]
            rhs[i] = 6.0 * (term1 - term2)

        # Resolucion mediante el Algoritmo de Thomas (eliminacion gaussiana tridiagonal)
        m_inner = self._thomas_algorithm(alpha, beta, gamma, rhs)

        # Ensamblar vector completo de momentos M_0, M_1, ..., M_n
        # Fronteras naturales: M_0 = 0, M_n = 0
        M = np.zeros(n + 1)
        M[1:n] = m_inner
        M[0] = 0.0
        M[n] = 0.0

        # Calculo de coeficientes polinomicos para cada intervalo
        for i in range(n):
            self.a[i] = y[i]
            self.b[i] = (y[i + 1] - y[i]) / h[i] - (h[i] / 6.0) * (2.0 * M[i] + M[i + 1])
            self.c[i] = M[i] / 2.0
            self.d[i] = (M[i + 1] - M[i]) / (6.0 * h[i])

    @staticmethod
    def _thomas_algorithm(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
        """
        Algoritmo de Thomas para resolver sistemas tridiagonales en tiempo O(N).
        """
        n = len(d)
        c_prime = np.zeros(n)
        d_prime = np.zeros(n)

        c_prime[0] = c[0] / b[0]
        d_prime[0] = d[0] / b[0]

        for i in range(1, n):
            denom = b[i] - a[i] * c_prime[i - 1]
            if i < n - 1:
                c_prime[i] = c[i] / denom
            d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denom

        x = np.zeros(n)
        x[-1] = d_prime[-1]
        for i in range(n - 2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i + 1]

        return x

    def evaluate(self, x_eval: np.ndarray) -> np.ndarray:
        """
        Evalua el spline en una serie de puntos continuos x_eval.
        """
        x_eval_arr = np.asarray(x_eval, dtype=np.float64)
        y_eval = np.zeros_like(x_eval_arr)

        for idx, val in enumerate(x_eval_arr):
            # Localizar el intervalo [x_i, x_{i+1}]
            if val <= self.x[0]:
                i = 0
            elif val >= self.x[-1]:
                i = self.n - 1
            else:
                i = np.searchsorted(self.x, val) - 1
                i = max(0, min(i, self.n - 1))

            dx = val - self.x[i]
            y_eval[idx] = self.a[i] + self.b[i] * dx + self.c[i] * (dx**2) + self.d[i] * (dx**3)

        # Las areas no pueden ser negativas fisicamente
        return np.maximum(0.0, y_eval)

