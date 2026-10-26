"""Генерация 20 измерений и множественная линейная регрессия."""

import csv
from pathlib import Path

import numpy as np
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results"
SEED = 18022004
SAMPLE_SIZE = 20
TRUE_COEFFICIENTS = np.array([1., 3., -2., 1.])


def generate_data(size=SAMPLE_SIZE, seed=SEED):
    if size <= 4:
        raise ValueError("Для четырёх коэффициентов и положительного числа степеней свободы нужно n>4")
    rng = np.random.RandomState(seed)
    factors = rng.normal(0, 1, (size, 3))
    noise = rng.normal(0, 1, size)
    response = 1 + 3 * factors[:, 0] - 2 * factors[:, 1] + factors[:, 2] + noise
    if len(np.unique(factors, axis=0)) != size:
        raise ValueError("Векторы факторов должны быть различными")
    return factors, noise, response


def create_model(size=SAMPLE_SIZE, seed=SEED):
    factors, _, response = generate_data(size, seed)
    design = sm.add_constant(factors, has_constant="add")
    if np.linalg.matrix_rank(design) != 4:
        raise ValueError("Матрица факторов не имеет полного ранга")
    return sm.OLS(response, design).fit(), size


def save_sample(model, filename):
    factors = model.model.exog[:, 1:]
    response = model.model.endog
    signal = model.model.exog @ TRUE_COEFFICIENTS
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("measurement", "x1", "x2", "x3", "noise", "y", "fitted_y", "residual"))
        for i in range(len(response)):
            writer.writerow((i + 1, *factors[i], response[i] - signal[i], response[i],
                             model.fittedvalues[i], model.resid[i]))


def main():
    model, size = create_model()
    print(f"n={size}, seed={SEED}")
    print(f"{'k':>3} {'x1':>10} {'x2':>10} {'x3':>10} {'y':>11}")
    for i, (row, y) in enumerate(zip(model.model.exog[:, 1:], model.model.endog), 1):
        print(f"{i:3d} {row[0]:10.6f} {row[1]:10.6f} {row[2]:10.6f} {y:11.6f}")
    print(model.summary())
    save_sample(model, OUTPUT / "model_sample.csv")
    print(f"Таблица измерений: {OUTPUT / 'model_sample.csv'}")


if __name__ == "__main__":
    main()
