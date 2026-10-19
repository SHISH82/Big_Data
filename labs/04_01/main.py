"""Монте-Карло: сравнение четырёх оценок параметра положения."""

import argparse
import csv
from pathlib import Path
import sys

import matplotlib

if "--save" in sys.argv or not sys.stdin.isatty():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
NUM_TRIALS = 10_000
SAMPLE_SIZE = 100
SEED = 18022004
HUBER_K = 1.44
ESTIMATORS = ("Выборочное среднее", "Выборочная медиана", "Хубер", "Двухэтапная")
SHORT_LABELS = ("Среднее", "Медиана", "Хубер", "Двухэтапная")
FIGURES = ("01_Дисперсии_оценок_Монте_Карло.png", "02_Распределения_оценок.png")


def huber_estimator(data, k=HUBER_K, tol=1e-6, max_iter=100):
    theta = float(np.median(data))
    scale = float(np.median(np.abs(data - theta)))
    if scale == 0:
        return theta
    for _ in range(max_iter):
        u = np.abs((data - theta) / scale)
        weights = np.ones_like(u)
        outside = u > k
        weights[outside] = k / u[outside]
        updated = float(np.sum(weights * data) / np.sum(weights))
        if abs(updated - theta) <= tol:
            return updated
        theta = updated
    raise RuntimeError("Оценка Хубера не сошлась за заданное число итераций")


def two_step_estimator(data):
    q1, q3 = np.percentile(data, [25, 75], method="linear")
    iqr = q3 - q1
    retained = data[(data >= q1 - 1.5 * iqr) & (data <= q3 + 1.5 * iqr)]
    if retained.size == 0:
        raise ValueError("После отбраковки не осталось наблюдений")
    return float(np.mean(retained))


def normal_distribution(rng, n):
    return rng.normal(0, 1, n)


def cauchy_distribution(rng, n):
    return rng.standard_cauchy(n)


def mixture_distribution(rng, n):
    # Каждое наблюдение выбирается из N(0,1) или C(0,1), а не складывается из них.
    normal_component = rng.rand(n) < 0.9
    return np.where(normal_component, rng.normal(0, 1, n), rng.standard_cauchy(n))


DISTRIBUTIONS = (
    ("N(0,1)", normal_distribution),
    ("C(0,1)", cauchy_distribution),
    ("0.9N(0,1) + 0.1C(0,1)", mixture_distribution),
)


def monte_carlo(distribution_func, n, num_trials, rng):
    estimates = np.empty((num_trials, len(ESTIMATORS)))
    for trial in range(num_trials):
        sample = distribution_func(rng, n)
        estimates[trial] = (
            np.mean(sample), np.median(sample),
            huber_estimator(sample), two_step_estimator(sample),
        )
    if not np.isfinite(estimates).all():
        raise FloatingPointError("Обнаружена неконечная оценка")
    return estimates


def summarise(estimates):
    # Формула лекции: делитель N — число испытаний Монте-Карло.
    means = np.array([np.mean(column) for column in estimates.T])
    variances = np.array([np.var(column, ddof=0) for column in estimates.T])
    return means, variances


def print_results(name, means, variances):
    print(f"\n{name}: {NUM_TRIALS} испытаний, n={SAMPLE_SIZE}")
    print(f"{'Оценка':<23} {'Среднее МК':>16} {'Дисперсия МК':>18}")
    for estimator, mean, variance in zip(ESTIMATORS, means, variances):
        print(f"{estimator:<23} {mean:>16.8g} {variance:>18.8g}")


def save_results(results, output):
    with (output / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("distribution", "estimator", "mean_mc", "variance_mc",
                         "num_trials", "sample_size", "seed", "huber_k"))
        for name, estimates in results.items():
            means, variances = summarise(estimates)
            for estimator, mean, variance in zip(ESTIMATORS, means, variances):
                writer.writerow((name, estimator, mean, variance, NUM_TRIALS,
                                 SAMPLE_SIZE, SEED, HUBER_K))


def draw_plots(results):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.grid": True, "grid.alpha": 0.2, "axes.axisbelow": True})
    colors = ("tab:red", "tab:blue", "tab:green", "tab:orange")
    variance_fig, variance_axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    distribution_fig, distribution_axes = plt.subplots(1, 3, figsize=(15, 6), layout="constrained")
    for ax, box_ax, (name, estimates) in zip(variance_axes, distribution_axes, results.items()):
        _, variances = summarise(estimates)
        bars = ax.bar(SHORT_LABELS, variances, color=colors)
        ax.set_yscale("log")
        ax.set(title=name, ylabel="Эмпирическая дисперсия (лог. шкала)")
        ax.tick_params(axis="x", labelrotation=25)
        ax.set_ylim(variances.min() / 2, variances.max() * 3)
        for bar, value in zip(bars, variances):
            ax.text(bar.get_x() + bar.get_width() / 2, value * 1.08,
                    f"{value:.4g}", ha="center", va="bottom", fontsize=9)

        boxes = box_ax.boxplot(estimates, patch_artist=True, whis=1.5,
                                flierprops={"markersize": 2, "alpha": 0.3})
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        box_ax.set_xticks(range(1, 5), SHORT_LABELS, rotation=25)
        box_ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        if name != "N(0,1)":
            box_ax.set_yscale("symlog", linthresh=0.5)
            box_ax.set_ylabel("Значение оценки (symlog, линейно при |x| ≤ 0.5)")
        else:
            box_ax.set_ylabel("Значение оценки")
        box_ax.set_title(name)
    variance_fig.suptitle("Дисперсии оценок по 10 000 испытаниям, n=100, seed=18022004")
    distribution_fig.suptitle("Распределения 10 000 оценок: точки за усами также показаны")
    return variance_fig, distribution_fig


def main():
    parser = argparse.ArgumentParser(description="Лабораторная 04_01: робастность оценок")
    parser.add_argument("--save", action="store_true", help="Сохранить результаты без открытия окон")
    args = parser.parse_args()
    rng = np.random.RandomState(SEED)
    results = {}
    print(f"Монте-Карло: N={NUM_TRIALS}, n={SAMPLE_SIZE}, seed={SEED}, k={HUBER_K}", flush=True)
    for name, distribution in DISTRIBUTIONS:
        print(f"Расчёт {name}...", flush=True)
        estimates = monte_carlo(distribution, SAMPLE_SIZE, NUM_TRIALS, rng)
        results[name] = estimates
        print_results(name, *summarise(estimates))
    print("\nДля Коши и смеси числа в строке среднего описывают только конечный эксперимент.")
    print("Теоретические математическое ожидание и дисперсия выборочного среднего не существуют.")
    output = ROOT / "results"
    output.mkdir(parents=True, exist_ok=True)
    save_results(results, output)
    figures = draw_plots(results)
    for figure, filename in zip(figures, FIGURES):
        figure.savefig(output / filename, dpi=160, bbox_inches="tight")
    print(f"Таблица и графики: {output}\nОбъяснения: {ROOT / 'README.md'}\nОтветы: {ROOT / 'answer.md'}")
    if not args.save and sys.stdin.isatty():
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
