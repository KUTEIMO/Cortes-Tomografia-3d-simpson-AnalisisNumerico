"""
Paquete del Motor Numerico de NeuroCalc
"""
from src.engine.simpson_engine import NumericalEngine, IntegrationResult
from src.engine.error_evaluator import ErrorEvaluator, BenchmarkComparison
from src.engine.spline_interpolator import NaturalCubicSpline
from src.engine.romberg_engine import RombergEngine

__all__ = [
    "NumericalEngine",
    "IntegrationResult",
    "ErrorEvaluator",
    "BenchmarkComparison",
    "NaturalCubicSpline",
    "RombergEngine"
]
