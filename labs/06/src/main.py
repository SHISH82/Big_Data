"""Полиномы степени 11: Ridge, Lasso, выбор штрафа и влияние шума."""

import argparse
import csv
from pathlib import Path
import sys

import matplotlib

if "--save" in sys.argv or not sys.stdin.isatty():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from base import ACTIVE_TOL, fit_model, get_coefficients, kkt_violation, loocv_mse
from plot import FIGURES, draw_plots


ROOT = Path(__file__).resolve().parents[1]
PENALTIES = np.logspace(-4, 2, 25)
SEED = 42
SIGMAS = (0., 0.1, 0.2, 0.3)


def load_points():
    with (ROOT / "data/points.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    x = np.array([float(row["x"]) for row in rows])
    y = np.array([float(row["y"]) for row in rows])
    if x.shape != (5,) or len(np.unique(x)) != 5 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Нужны пять различных конечных точек из условия")
    return x, y


def save_table(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def print_table(headers, rows):
    cells = [[str(value) for value in row] for row in [headers, *rows]]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]

    def line(row):
        return " | ".join(value.ljust(width) if i == 0 else value.rjust(width)
                          for i, (value, width) in enumerate(zip(row, widths)))

    print(line(cells[0]))
    print("-+-".join("-" * width for width in widths))
    for row in cells[1:]:
        print(line(row))


def format_active_terms(value):
    powers = [int(power) for power in value.split(";") if power]
    if len(powers) >= 3 and powers == list(range(powers[0], powers[-1] + 1)):
        return f"{powers[0]}–{powers[-1]}"
    return ", ".join(map(str, powers)) or "нет"


def print_results(selected, models):
    print("\nРЕЗУЛЬТАТЫ: МОДЕЛИ С МИНИМАЛЬНОЙ MSE LOOCV")
    methods = ("Ridge", "Lasso")
    for sigma in SIGMAS:
        rows = [selected[(sigma, method)] for method in methods]
        print("\n" + "=" * 72)
        print(f"УРОВЕНЬ ШУМА: σ = {sigma}")
        print("=" * 72)
        positions = ["нижняя граница" if row["lambda"] == float(PENALTIES[0])
                     else "верхняя граница" if row["lambda"] == float(PENALTIES[-1])
                     else "внутри сетки" for row in rows]
        print_table(["Показатель", *methods], [
            ["λ", *(row["lambda"] for row in rows)],
            ["MSE LOOCV", *(row["loocv_mse"] for row in rows)],
            ["RSS", *(row["RSS"] for row in rows)],
            ["Ненулевые степени", *(format_active_terms(row["active_terms"]) for row in rows)],
            ["Положение λ", *positions],
        ])
        beta = [get_coefficients(models[(sigma, method, row["lambda"])])[0]
                for method, row in zip(methods, rows)]
        print("\nКоэффициенты исходного полинома (β0 — свободный член):")
        print_table(["Коэффициент", *methods],
                    [[f"β{power}", *(values[power] for values in beta)] for power in range(12)])
    print("\nГраничный минимум — лучшее λ на краю проверенной сетки, а не внутренний оптимум.")


def run_experiment(x, clean):
    standard_noise = np.random.RandomState(SEED).normal(size=len(x))
    samples = {sigma: clean + sigma * standard_noise for sigma in SIGMAS}
    grid = np.linspace(x.min(), x.max(), 401)
    reference = 1 + grid ** 3
    models, selected, ols_models = {}, {}, {}
    metrics, coefficients = [], []
    for step, (sigma, y) in enumerate(samples.items(), start=1):
        print(f"[{step}/{len(samples)}] Расчёт моделей: σ = {sigma}", flush=True)
        ols_models[sigma] = fit_model("OLS", 0, x, y)
        for method in ("Ridge", "Lasso"):
            candidates = []
            for value in PENALTIES:
                penalty = float(value)
                model = fit_model(method, penalty, x, y)
                models[(sigma, method, penalty)] = model
                beta, theta = get_coefficients(model)
                active = np.flatnonzero(np.abs(theta[1:]) > ACTIVE_TOL) + 1
                residual = y - np.asarray(model.predict(x[:, None])).ravel()
                row = {"sigma": sigma, "method": method, "lambda": penalty,
                       "sklearn_alpha": penalty if method == "Ridge" else penalty / (2 * len(x)),
                       "RSS": float(residual @ residual),
                       "R_squared": float(1 - residual @ residual / np.sum((y - y.mean()) ** 2)),
                       "loocv_mse": loocv_mse(method, penalty, x, y),
                       "reference_mse": float(np.mean((np.asarray(model.predict(grid[:, None])).ravel() - reference) ** 2)),
                       "active_terms": ";".join(map(str, active)), "n_active": len(active),
                       "effective_degree": int(active.max()) if len(active) else 0,
                       "coefficient_shift_l2": float(np.linalg.norm(beta - get_coefficients(models[(0., method, penalty)])[0])),
                       "kkt_violation": kkt_violation(model, method, penalty, x, y), "selected": False}
                metrics.append(row)
                candidates.append(row)
                for power in range(12):
                    coefficients.append({"sigma": sigma, "method": method, "lambda": penalty,
                                         "power": power, "coefficient": beta[power],
                                         "standardized_coefficient": theta[power]})
            best = min(candidates, key=lambda r: (r["loocv_mse"], -r["lambda"]))
            best["selected"] = True
            selected[(sigma, method)] = best
    return samples, models, metrics, coefficients, selected, ols_models


def main():
    parser = argparse.ArgumentParser(description="Лабораторная 06: гребневая и лассо-регрессия")
    parser.add_argument("--save", action="store_true", help="Сохранить графики без открытия окон")
    args = parser.parse_args()
    x, clean = load_points()
    samples, models, metrics, coefficients, selected, ols_models = run_experiment(x, clean)
    output = ROOT / "results"
    output.mkdir(parents=True, exist_ok=True)
    save_table(output / "metrics.csv", metrics)
    save_table(output / "coefficients.csv", coefficients)
    save_table(output / "selected_models.csv", list(selected.values()))
    save_table(output / "samples.csv",
               [{"sigma": sigma, "x": float(xi), "y_clean": float(yi), "noise": float(noisy - yi),
                 "y_observed": float(noisy)} for sigma, y in samples.items() for xi, yi, noisy in zip(x, clean, y)])
    print_results(selected, models)
    figures = draw_plots(x, samples, PENALTIES, models, metrics, selected, ols_models)
    for figure, filename in zip(figures, FIGURES):
        figure.savefig(output / filename, dpi=160, bbox_inches="tight")
    print(f"\nТаблицы и графики: {output}\nСтрогий вывод: {ROOT / 'formula.md'}\nОтчёт: {ROOT / 'README.md'}")
    if not args.save and sys.stdin.isatty():
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
