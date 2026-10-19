"""Мультипликативная модель веса от возраста через логарифмирование."""

from dataclasses import dataclass

import numpy as np

from part1 import FitQuality, LinearResult, build_linear_regression, evaluate_fit, validate_pairs


@dataclass
class MultiplicativeResult:
    alpha: float
    gamma: float
    log_model: LinearResult
    fitted: np.ndarray
    quality: FitQuality
    correction: float
    corrected_fitted: np.ndarray
    corrected_quality: FitQuality


def build_multiplicative_model(age, weight):
    age, weight = validate_pairs(age, weight)
    if np.any(age <= 0) or np.any(weight <= 0):
        raise ValueError("Для логарифмирования возраст и вес должны быть положительными")
    log_model = build_linear_regression(np.log(age), np.log(weight))
    alpha = float(np.exp(log_model.intercept))
    gamma = log_model.slope
    fitted = np.exp(log_model.fitted)
    # Поправка обратного преобразования: среднее относительных остатков равно 1.
    correction = float(np.mean(np.exp(log_model.residuals)))
    corrected = correction * fitted
    return MultiplicativeResult(alpha, gamma, log_model, fitted, evaluate_fit(weight, fitted),
                                correction, corrected, evaluate_fit(weight, corrected))
