"""Простая линейная регрессия и показатели качества модели."""

from dataclasses import dataclass

import numpy as np


@dataclass
class FitQuality:
    rss: float
    rse: float
    r2: float
    tss: float


@dataclass
class LinearResult:
    intercept: float
    slope: float
    fitted: np.ndarray
    residuals: np.ndarray
    quality: FitQuality


def validate_pairs(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size < 3:
        raise ValueError("Нужны два одномерных массива одинаковой длины n>2")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("В данных не должно быть пропусков и бесконечных значений")
    if np.all(x == x[0]):
        raise ValueError("Все значения фактора одинаковы: наклон не определён")
    return x, y


def evaluate_fit(y, fitted):
    residuals = y - fitted
    rss = float(np.sum(residuals ** 2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    return FitQuality(rss, float(np.sqrt(rss / (len(y) - 2))),
                      1 - rss / tss if tss > 0 else float("nan"), tss)


def build_linear_regression(x, y):
    x, y = validate_pairs(x, y)
    mean_x, mean_y = np.mean(x), np.mean(y)
    slope = float(np.sum((x - mean_x) * (y - mean_y)) / np.sum((x - mean_x) ** 2))
    intercept = float(mean_y - slope * mean_x)
    fitted = intercept + slope * x
    return LinearResult(intercept, slope, fitted, y - fitted, evaluate_fit(y, fitted))
