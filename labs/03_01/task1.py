"""Задание 1: модельный ряд и метод Прони."""

import argparse
from pathlib import Path
import sys

import matplotlib

if "--save" in sys.argv or not sys.stdin.isatty():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results"

FIGURE = "01_Метод_Прони_и_ошибка_восстановления.png"
FREQUENCY_FIGURE = "05_Колебание_f4_на_модельном_ряде.png"
FREQUENCY_TEXT = "Частоты_метода_Прони.txt"


def get_model_row(n=200, h=0.02):
    """Модель из условия: i=1,...,n, три затухающих косинусоиды."""
    time = np.arange(1, n + 1) * h
    model = sum(k * np.exp(-time / k) * np.cos(4 * np.pi * k * time + np.pi / k)
                for k in range(1, 4))
    return time, model


def prony(x, order=6, step=0.02):
    """Разложение вещественного ряда в сумму комплексных экспонент."""
    x = np.asarray(x, dtype=float)
    if (x.ndim != 1 or not np.isfinite(x).all() or step <= 0
            or not isinstance(order, int) or order < 1 or len(x) < 2 * order):
        raise ValueError("Нужны конечный одномерный ряд, n >= 2m, целый m > 0 и h > 0")
    rows = np.arange(order, len(x))
    # x[i] + a[0]*x[i-1] + ... + a[m-1]*x[i-m] = 0.
    lag_matrix = np.column_stack([x[rows - j] for j in range(1, order + 1)])
    a, _, rank, _ = np.linalg.lstsq(lag_matrix, -x[rows], rcond=None)
    if rank < order:
        raise ValueError("Недостаточный ранг: уменьшите порядок метода Прони")
    poles = np.roots(np.r_[1, a])
    if np.any(np.abs(poles) < 1e-12):
        raise ValueError("Нулевой корень не позволяет определить затухание")
    # Как в лекции, первая степень равна нулю.
    vandermonde = poles[np.newaxis, :] ** np.arange(len(x))[:, np.newaxis]
    coefficients = np.linalg.lstsq(vandermonde, x, rcond=None)[0]
    reconstructed = (vandermonde @ coefficients).real

    # Первая точка модели соответствует t=h: переносим коэффициенты к t=0.
    initial_coefficients = coefficients / poles
    positive = np.flatnonzero(poles.imag > 1e-8)
    positive = positive[np.argsort(np.angle(poles[positive]))]
    parameters = pd.DataFrame({
        "amplitude": 2 * np.abs(initial_coefficients[positive]),
        "decay": np.log(np.abs(poles[positive])) / step,
        "frequency": np.angle(poles[positive]) / (2 * np.pi * step),
        "phase": np.angle(initial_coefficients[positive]),
    })
    return parameters, reconstructed


def draw_plot(time, model, parameters, reconstructed):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.grid": True, "grid.alpha": 0.25})
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), layout="constrained")
    axes[0].plot(time, model, color="black", label="Исходный ряд")
    axes[0].plot(time, reconstructed, "--", color="tab:red", label="Метод Прони")
    axes[0].set(title="Модельный ряд и восстановление методом Прони", ylabel="x(t)")
    axes[0].legend()
    for p in parameters.itertuples():
        component = p.amplitude * np.exp(p.decay * time) * np.cos(2 * np.pi * p.frequency * time + p.phase)
        axes[1].plot(time, component, label=f"f = {p.frequency:.0f}")
    axes[1].set(title="Три восстановленных затухающих колебания", ylabel="Компонента")
    axes[1].legend()
    axes[2].plot(time, reconstructed - model)
    axes[2].axhline(0, color="gray", linestyle="--")
    axes[2].set(title="Ошибка восстановления", ylabel="x̂ − x")
    for ax in axes:
        ax.set_xlabel("Время t = i · h")
    return fig


def draw_frequency_plot(time, model, parameters, frequency=4):
    selected = parameters.loc[np.isclose(parameters.frequency, frequency)]
    if selected.empty:
        raise ValueError(f"Не найдено восстановленное колебание с частотой f={frequency}")
    p = selected.iloc[0]
    component = p.amplitude * np.exp(p.decay * time) * np.cos(
        2 * np.pi * p.frequency * time + p.phase
    )
    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    ax.plot(time, model, color="tab:blue", linewidth=1.5,
            label="Исходный модельный ряд: сумма трёх колебаний")
    ax.plot(time, component, color="tab:orange", linewidth=2,
            label=f"Восстановленное колебание Прони: f = {p.frequency:.6f}")
    ax.set(title=f"Затухающее колебание f = {frequency:g} на модельном ряде",
           xlabel="Время t = i · h", ylabel="Значение")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.legend(loc="upper right")
    found_frequencies = "; ".join(f"{f:.6f}" for f in parameters.frequency)
    ax.text(0.98, 0.04,
            f"Найденные методом Прони частоты:\nf = {found_frequencies}\n"
            f"циклов / ед. времени\nНаложена компонента f = {p.frequency:.6f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "white",
                  "edgecolor": "gray", "alpha": 0.95})
    return fig


def run(save_only=False, verbose=True):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    time, model = get_model_row()
    parameters, reconstructed = prony(model, order=6, step=0.02)
    frequency_lines = [
        "Положительные частоты, найденные методом Прони в модельном ряде",
        "Единицы: циклы на единицу времени.",
        "Рассчитанные оценки приведены с 17 значащими цифрами.",
        "Небольшие отличия от теоретических 2, 4 и 6 обусловлены численной погрешностью.",
        "",
    ]
    frequency_lines.extend(
        f"f{i} = {value:.17g}"
        for i, value in enumerate(parameters.frequency, start=1)
    )
    (OUTPUT / FREQUENCY_TEXT).write_text("\n".join(frequency_lines) + "\n", encoding="utf-8")
    fig = draw_plot(time, model, parameters, reconstructed)
    fig.savefig(OUTPUT / FIGURE, dpi=160, bbox_inches="tight")
    frequency_fig = draw_frequency_plot(time, model, parameters, frequency=4)
    frequency_fig.savefig(OUTPUT / FREQUENCY_FIGURE, dpi=160, bbox_inches="tight")
    if verbose:
        print("Параметры трёх восстановленных колебаний:")
        print(parameters.rename(columns={
            "amplitude": "Амплитуда", "decay": "Затухание",
            "frequency": "Частота", "phase": "Фаза",
        }).to_string(index=False))
        print(f"MSE: {np.mean((reconstructed-model)**2):.3e}")
        print(f"RMSE: {np.sqrt(np.mean((reconstructed-model)**2)):.3e}")
        print(f"График: {OUTPUT / FIGURE}")
        print(f"Колебание f=4 на модельном ряде: {OUTPUT / FREQUENCY_FIGURE}")
        print(f"Рассчитанные частоты: {OUTPUT / FREQUENCY_TEXT}")
    if not save_only:
        plt.show()
    plt.close(fig)
    plt.close(frequency_fig)
    return model, parameters, reconstructed


def main():
    parser = argparse.ArgumentParser(description="Задание 1. Метод Прони")
    parser.add_argument("--save", action="store_true", help="Сохранить результат без окна графика")
    args = parser.parse_args()
    run(save_only=args.save or not sys.stdin.isatty())


if __name__ == "__main__":
    main()
