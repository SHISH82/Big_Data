"""Правило трёх сигм и боксплот Тьюки для модельного ряда."""

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

if "--save" in sys.argv or not sys.stdin.isatty():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ADDITIONAL = np.array([5.0, -4.0, 3.3, 2.99, -3.0])
FIGURES = (
    "01_Правило_трех_сигм_для_крайних_значений.png",
    "02_Боксплот_Тьюки_и_сравнение_аномалий.png",
)


@dataclass
class SigmaRuleResult:
    mean: float
    std: float
    lower_bound: float
    upper_bound: float
    ranks: np.ndarray
    values: np.ndarray
    is_anomaly: np.ndarray

    @property
    def anomalies(self):
        return self.values[self.is_anomaly]


@dataclass
class TukeyResult:
    q1: float
    median: float
    q3: float
    iqr: float
    lower_bound: float
    upper_bound: float
    is_anomaly: np.ndarray
    anomalies: np.ndarray


def generate_data(seed=42):
    # RandomState воспроизводит последовательность np.random.seed из примера.
    normal = np.random.RandomState(seed).normal(0, 1, 195)
    data = np.concatenate((normal, ADDITIONAL))
    return data, np.sort(data)


def apply_three_sigma(data):
    sorted_data = np.sort(data)
    mean = float(np.mean(data))
    std = float(np.std(data, ddof=0))
    lower, upper = mean - 3 * std, mean + 3 * std
    indices = np.r_[0:3, len(data) - 3:len(data)]
    values = sorted_data[indices]
    return SigmaRuleResult(mean, std, lower, upper, indices + 1, values,
                           (values < lower) | (values > upper))


def apply_tukey(data):
    q1, median, q3 = np.percentile(data, [25, 50, 75], method="linear")
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (data < lower) | (data > upper)
    return TukeyResult(q1, median, q3, iqr, lower, upper, mask, np.sort(data[mask]))


def print_results(data, sorted_data, sigma, tukey):
    np.set_printoptions(precision=6, suppress=True)
    print(f"N(0,1): 195 значений; добавлены {ADDITIONAL}; всего {len(data)}. Seed=42.")
    print("\nОтсортированный ряд:")
    print(sorted_data)
    print(f"\nСреднее: {sigma.mean:.6f}; s (ddof=0): {sigma.std:.6f}")
    print(f"Границы трёх сигм: [{sigma.lower_bound:.6f}; {sigma.upper_bound:.6f}]")
    print("Проверка первых трёх и последних трёх порядковых статистик:")
    for rank, value, anomaly in zip(sigma.ranks, sigma.values, sigma.is_anomaly):
        print(f"  x({rank:3d}) = {value: .6f}; |x−среднее|/s = {abs(value-sigma.mean)/sigma.std:.6f}; "
              f"{'аномалия' if anomaly else 'не аномалия'}")
    print(f"\nQ1={tukey.q1:.6f}; медиана={tukey.median:.6f}; Q3={tukey.q3:.6f}; IQR={tukey.iqr:.6f}")
    print(f"Границы Тьюки: [{tukey.lower_bound:.6f}; {tukey.upper_bound:.6f}]")
    print("Аномалии по трём сигмам среди шести проверенных:", sigma.anomalies)
    print("Все аномалии Тьюки:", tukey.anomalies)
    print("Оба метода:", np.intersect1d(sigma.anomalies, tukey.anomalies))
    print("Только Тьюки:", np.setdiff1d(tukey.anomalies, sigma.anomalies))
    print("Только три сигмы:", np.setdiff1d(sigma.anomalies, tukey.anomalies))
    print("\nПроисхождение выбросов Тьюки:")
    for index in np.flatnonzero(tukey.is_anomaly):
        origin = "добавленное значение" if index >= 195 else "из N(0,1)"
        print(f"  x[{index+1}] = {data[index]: .6f}: {origin}")


def draw_sigma_plot(sorted_data, sigma):
    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    ranks = np.arange(1, len(sorted_data) + 1)
    ax.axhspan(sigma.lower_bound, sigma.upper_bound, color="tab:green", alpha=0.08)
    ax.plot(ranks, sorted_data, ".-", markersize=3, color="gray", label="Отсортированный ряд")
    ax.axhline(sigma.mean, color="tab:green", linestyle=":", label=f"Среднее: {sigma.mean:.3f}")
    ax.axhline(sigma.lower_bound, color="tab:red", linestyle="--",
               label=f"Границы 3s: {sigma.lower_bound:.3f} и {sigma.upper_bound:.3f}")
    ax.axhline(sigma.upper_bound, color="tab:red", linestyle="--")
    for mask, color, label in ((sigma.is_anomaly, "tab:red", "Аномалии среди проверенных"),
                               (~sigma.is_anomaly, "tab:blue", "Проверены, не аномалии")):
        ax.scatter(sigma.ranks[mask], sigma.values[mask], s=70, color=color, zorder=3, label=label)
    offsets = ((25, -20), (45, 8), (85, 34), (-125, -40), (-110, 5), (-105, 5))
    for rank, value, offset in zip(sigma.ranks, sigma.values, offsets):
        ax.annotate(f"x({rank}) = {value:.3f}", (rank, value), xytext=offset,
                    textcoords="offset points", fontsize=10,
                    arrowprops={"arrowstyle": "-", "color": "gray"})
    ax.set(title="Правило трёх сигм: проверка шести крайних порядковых статистик",
           xlabel="Порядковый номер после сортировки", ylabel="Значение", ylim=(-4.8, 5.8))
    ax.legend(loc="upper left", fontsize=9)
    return fig


def draw_tukey_plot(data, sigma, tukey):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), layout="constrained",
                             gridspec_kw={"height_ratios": [1, 2]})
    box = axes[0].boxplot(data, vert=False, whis=1.5, patch_artist=True,
                         boxprops={"facecolor": "#c7deed"},
                         flierprops={"marker": "o", "markeredgecolor": "tab:red"})
    axes[0].axvline(tukey.lower_bound, color="tab:orange", linestyle="--",
                   label=f"Границы Тьюки: {tukey.lower_bound:.3f} и {tukey.upper_bound:.3f}")
    axes[0].axvline(tukey.upper_bound, color="tab:orange", linestyle="--")
    axes[0].set(title="Боксплот Тьюки: 1,5 · IQR, выбросы отмечены кружками", xlabel="Значение")
    axes[0].set_yticks([])
    axes[0].legend(loc="upper left", fontsize=9)

    order = np.argsort(data, kind="stable")
    ranks = np.arange(1, len(data) + 1)
    sorted_data = data[order]
    sigma_mask = np.zeros(len(data), dtype=bool)
    sigma_mask[sigma.ranks - 1] = sigma.is_anomaly
    tukey_mask = tukey.is_anomaly[order]
    axes[1].scatter(ranks, sorted_data, s=13, color="gray", label="Все наблюдения")
    for mask, color, marker, label in (
        (sigma_mask & tukey_mask, "tab:red", "o", "Аномалии обоих методов"),
        (tukey_mask & ~sigma_mask, "tab:orange", "D", "Аномалии только Тьюки"),
        (sigma_mask & ~tukey_mask, "tab:blue", "s", "Аномалии только трёх сигм"),
    ):
        if mask.any():
            axes[1].scatter(ranks[mask], sorted_data[mask], s=55, color=color, marker=marker, label=label)
    for bounds, color, style, label in (
        ((sigma.lower_bound, sigma.upper_bound), "tab:red", "--", "Границы трёх сигм"),
        ((tukey.lower_bound, tukey.upper_bound), "tab:orange", ":", "Границы Тьюки"),
    ):
        axes[1].axhline(bounds[0], color=color, linestyle=style, label=label)
        axes[1].axhline(bounds[1], color=color, linestyle=style)
    axes[1].set(title="Сравнение методов на одном и том же ряде",
                xlabel="Порядковый номер после сортировки", ylabel="Значение")
    axes[1].legend(loc="upper left", fontsize=9)
    return fig, np.sort(box["fliers"][0].get_xdata())


def main():
    parser = argparse.ArgumentParser(description="Лабораторная 03_02: обнаружение аномалий")
    parser.add_argument("--save", action="store_true", help="Сохранить графики без открытия окон")
    args = parser.parse_args()
    data, sorted_data = generate_data()
    sigma = apply_three_sigma(data)
    tukey = apply_tukey(data)
    print_results(data, sorted_data, sigma, tukey)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.grid": True, "grid.alpha": 0.2})
    first = draw_sigma_plot(sorted_data, sigma)
    second, _ = draw_tukey_plot(data, sigma, tukey)
    output = ROOT / "results"
    output.mkdir(parents=True, exist_ok=True)
    for figure, filename in zip((first, second), FIGURES):
        figure.savefig(output / filename, dpi=160, bbox_inches="tight")
    print(f"\nГрафики: {output}\nОтчёт: {ROOT / 'README.md'}")
    if not args.save and sys.stdin.isatty():
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
