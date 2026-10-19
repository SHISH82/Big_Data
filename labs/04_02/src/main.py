"""Линейная и мультипликативная регрессия на учебных данных из примера."""

import argparse
import csv
from pathlib import Path
import sys

import matplotlib

if "--save" in sys.argv or not sys.stdin.isatty():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from part1 import build_linear_regression
from part2 import build_multiplicative_model


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ("01_Линейная_регрессия_веса_от_роста.png",
           "02_Мультипликативная_модель_веса_от_возраста.png")


def load_table(filename, factor, min_size, max_size):
    with filename.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not min_size <= len(rows) <= max_size:
        raise ValueError(f"В {filename.name} должно быть от {min_size} до {max_size} строк")
    identifiers = [row["person"] for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"Повторяющиеся идентификаторы в {filename.name}")
    x = np.array([float(row[factor]) for row in rows])
    y = np.array([float(row["weight_kg"]) for row in rows])
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("В таблице нужны конечные положительные значения")
    return x, y


def print_quality(label, quality):
    print(f"{label}: RSS={quality.rss:.9f}; RSE={quality.rse:.9f}; R²={quality.r2:.9f}")


def draw_linear_plot(height, weight, result):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), layout="constrained",
                              gridspec_kw={"height_ratios": [2, 1]})
    grid = np.linspace(height.min(), height.max(), 200)
    axes[0].scatter(height, weight, label=f"Учебные наблюдения, n={len(height)}", zorder=3)
    axes[0].plot(grid, result.intercept + result.slope * grid, color="tab:red",
                 label=f"Вес = {result.intercept:.4f} + {result.slope:.4f} · рост")
    axes[0].set(title="Линейная регрессия веса от роста", xlabel="Рост, см", ylabel="Вес, кг")
    axes[0].legend(loc="lower right")
    axes[0].text(0.03, 0.95,
                 f"RSS = {result.quality.rss:.4f} кг²\nRSE = {result.quality.rse:.4f} кг\nR² = {result.quality.r2:.6f}",
                 transform=axes[0].transAxes, va="top",
                 bbox={"facecolor": "white", "edgecolor": "gray", "alpha": 0.9})
    axes[1].scatter(height, result.residuals)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set(title="Остатки линейной модели", xlabel="Рост, см", ylabel="Вес − прогноз, кг")
    return fig


def draw_multiplicative_plot(age, weight, result):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), layout="constrained",
                              gridspec_kw={"height_ratios": [2, 1]})
    grid = np.linspace(age.min(), age.max(), 300)
    curve = result.alpha * grid ** result.gamma
    axes[0].scatter(age, weight, label=f"Учебные наблюдения, n={len(age)}", zorder=3)
    axes[0].plot(grid, curve, color="tab:red", linewidth=2,
                 label=f"Вес = {result.alpha:.6f} · возраст^{result.gamma:.6f}")
    axes[0].plot(grid, result.correction * curve, color="tab:green", linestyle="--",
                 label="С поправкой обратного преобразования")
    axes[0].set(title="Мультипликативная регрессия веса от возраста",
                xlabel="Возраст, лет", ylabel="Вес, кг")
    axes[0].legend(loc="lower right")
    axes[0].text(0.03, 0.95,
                 f"RSS = {result.quality.rss:.4f} кг²\nRSE = {result.quality.rse:.4f} кг\nR² (кг) = {result.quality.r2:.6f}\nR² (логарифмы) = {result.log_model.quality.r2:.6f}",
                 transform=axes[0].transAxes, va="top",
                 bbox={"facecolor": "white", "edgecolor": "gray", "alpha": 0.9})
    axes[1].scatter(age, weight - result.fitted)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set(title="Остатки базовой степенной модели в исходных единицах",
                xlabel="Возраст, лет", ylabel="Вес − прогноз, кг")
    return fig


def save_metrics(output, linear, nonlinear):
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("model", "scale", "intercept_or_alpha", "slope_or_gamma", "n",
                         "RSS", "RSE", "R_squared"))
        for name, scale, intercept, slope, n, quality in (
            ("linear_height_weight", "kg", linear.intercept, linear.slope, len(linear.fitted), linear.quality),
            ("log_age_weight", "ln(kg)", nonlinear.log_model.intercept, nonlinear.gamma, len(nonlinear.fitted), nonlinear.log_model.quality),
            ("power_age_weight", "kg", nonlinear.alpha, nonlinear.gamma, len(nonlinear.fitted), nonlinear.quality),
            ("power_age_weight_corrected", "kg", nonlinear.alpha * nonlinear.correction,
             nonlinear.gamma, len(nonlinear.fitted), nonlinear.corrected_quality),
        ):
            writer.writerow((name, scale, intercept, slope, n, quality.rss, quality.rse, quality.r2))


def main():
    parser = argparse.ArgumentParser(description="Лабораторная 04_02: регрессионный анализ")
    parser.add_argument("--save", action="store_true", help="Сохранить графики без открытия окон")
    args = parser.parse_args()
    height, weight = load_table(ROOT / "data/height_weight.csv", "height_cm", 5, 10)
    age, age_weight = load_table(ROOT / "data/age_weight.csv", "age_years", 10, 20)
    linear = build_linear_regression(height, weight)
    nonlinear = build_multiplicative_model(age, age_weight)
    print("Данные синтетические, получены преобразованием генератора из примера (seed=42).")
    print(f"Линейная модель, n={len(height)}: вес = {linear.intercept:.9f} + {linear.slope:.9f} · рост")
    print(f"Σx={height.sum():g}; Σy={weight.sum():g}; Σx²={np.sum(height**2):g}; Σxy={np.sum(height*weight):g}")
    print_quality("Линейная модель (кг)", linear.quality)
    print(f"\nМультипликативная модель, n={len(age)}: вес = {nonlinear.alpha:.9f} · возраст^{nonlinear.gamma:.9f}")
    print_quality("Логарифмическая шкала", nonlinear.log_model.quality)
    print_quality("Исходная шкала (кг)", nonlinear.quality)
    print(f"Поправка обратного преобразования C={nonlinear.correction:.9f}")
    print_quality("Исходная шкала с поправкой (кг)", nonlinear.corrected_quality)
    output = ROOT / "results"
    output.mkdir(parents=True, exist_ok=True)
    save_metrics(output, linear, nonlinear)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.grid": True, "grid.alpha": 0.2})
    figures = (draw_linear_plot(height, weight, linear), draw_multiplicative_plot(age, age_weight, nonlinear))
    for figure, filename in zip(figures, FIGURES):
        figure.savefig(output / filename, dpi=160, bbox_inches="tight")
    print(f"\nГрафики и показатели: {output}\nРучная выкладка и отчёт: {ROOT / 'README.md'}")
    if not args.save and sys.stdin.isatty():
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
