"""Задание 2: тренд и сезонность температуры Воронежа."""

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

from statsmodels.tsa.seasonal import seasonal_decompose
from weather_parser import YEARS, load_weather

FIGURES = (
    "01_Метод_Прони_и_ошибка_восстановления.png",
    "02_Температура_Воронежа_и_сглаживание.png",
    "03_Тренд_сезонность_и_остатки.png",
    "04_Годовая_сезонность_и_спектр.png",
)
MONTHS = ("Январь", "Февраль", "Март", "Апрель", "Май", "Июнь",
          "Июль", "Август", "Сентябрь", "Октябрь", "Ноябрь", "Декабрь")


def analyse_weather(data):
    series = data.set_index("date")["temperature"].asfreq("D")
    decomposition = seasonal_decompose(series, model="additive", period=365)
    trend = decomposition.trend.dropna()
    years = (trend.index - series.index[0]).days.to_numpy() / 365.2425
    slope = np.polyfit(years, trend.to_numpy(), 1)[0]
    monthly = data.pivot_table(index=data.date.dt.month, columns=data.date.dt.year,
                              values="temperature", aggfunc="mean")
    annual = data.groupby(data.date.dt.year)["temperature"].mean()
    # Убираем постоянную составляющую; окно Ханна снижает утечку спектра.
    window = np.hanning(len(series))
    spectrum = np.fft.rfft((series.to_numpy() - series.mean()) * window)
    amplitude = 2 * np.abs(spectrum) / window.sum()
    amplitude[0] /= 2
    if len(series) % 2 == 0:
        amplitude[-1] /= 2
    frequency = np.fft.rfftfreq(len(series), d=1)
    peak = int(np.argmax(amplitude[1:]) + 1)
    return {"series": series, "decomposition": decomposition, "slope": slope,
            "monthly": monthly, "annual": annual, "frequency": frequency,
            "amplitude": amplitude, "peak": peak}


def draw_temperature_plot(analysis):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.grid": True, "grid.alpha": 0.25})
    series, dec = analysis["series"], analysis["decomposition"]
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    ax.plot(series.index, series, color="gray", alpha=0.6, label="Среднесуточная температура")
    ax.plot(series.index, series.rolling(31, center=True).mean(), label="Среднее за 31 день")
    ax.plot(series.index, dec.trend, color="tab:red", linewidth=2.5, label="Тренд: среднее за 365 дней")
    ax.set(title="Воронеж, Россия: температура за 2024–2025 годы", xlabel="Дата", ylabel="Температура, °C")
    ax.legend(loc="lower right")
    return fig


def draw_decomposition_plot(analysis):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.grid": True, "grid.alpha": 0.25})
    series, dec = analysis["series"], analysis["decomposition"]
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True, layout="constrained")
    for ax, values, title, color in zip(axes,
            (series, dec.trend, dec.seasonal, dec.resid),
            ("Исходный ряд", "Тренд: центрированное среднее за 365 дней",
             "Сезонная компонента: период 365 дней", "Остатки: ряд − тренд − сезонность"),
            ("gray", "tab:red", "tab:green", "tab:blue")):
        ax.plot(series.index, values, color=color)
        ax.set(title=title, ylabel="°C")
    axes[-1].set_xlabel("Дата")
    fig.suptitle("Воронеж: аддитивное разложение температуры")
    return fig


def draw_spectrum_plot(analysis):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.grid": True, "grid.alpha": 0.25})
    series, dec = analysis["series"], analysis["decomposition"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), layout="constrained")
    for year in YEARS:
        axes[0].plot(range(1, 13), analysis["monthly"][year], "o-", label=str(year))
    axes[0].set_xticks(range(1, 13), [m[:3] for m in MONTHS])
    axes[0].set(title="Годовой цикл: среднемесячные температуры", ylabel="Температура, °C", xlabel="Месяц")
    axes[0].legend()
    frequency, amplitude, peak = analysis["frequency"], analysis["amplitude"], analysis["peak"]
    low = frequency <= 0.05
    axes[1].plot(frequency[low], amplitude[low])
    axes[1].axvline(1 / 365.2425, color="gray", linestyle="--", label="Годовая частота 1 / 365.2425")
    axes[1].plot(frequency[peak], amplitude[peak], "ro", label=f"Главный пик: период {1 / frequency[peak]:.1f} суток")
    axes[1].set(title="Спектр Фурье: низкие частоты, окно Ханна", xlabel="Частота, циклов/сутки", ylabel="Амплитуда, °C")
    axes[1].legend()
    return fig


def draw_original_plot(analysis):
    series = analysis["series"]
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    ax.plot(series.index, series)
    ax.set(title="Воронеж: исходные среднесуточные температуры за 2024–2025 годы",
           xlabel="Дата", ylabel="Температура, °C")
    ax.grid(alpha=0.25)
    return fig


def print_results(analysis):
    print("Среднегодовые температуры Воронежа:")
    for year, value in analysis["annual"].items():
        print(f"  {year}: {value:.2f} °C")
    print(f"Наклон сглаженного тренда: {analysis['slope']:+.3f} °C/год (описательная оценка)")
    print("Двух лет недостаточно для вывода о долговременном климатическом тренде.")
    seasonal = analysis["decomposition"].seasonal
    seasonal_months = seasonal.groupby(seasonal.index.month).mean()
    print("\nСредняя сезонная компонента по месяцам:")
    for number, value in seasonal_months.items():
        print(f"  {MONTHS[number-1]:<10} {value:+.2f} °C")
    peak_frequency = analysis["frequency"][analysis["peak"]]
    print(f"Главная частота: {peak_frequency:.8f} циклов/сутки, период {1/peak_frequency:.1f} суток")


def menu(analysis):
    actions = {"0": draw_original_plot, "1": draw_decomposition_plot,
               "2": draw_temperature_plot, "3": draw_spectrum_plot}
    prompt = (
        "\n0 — Исходная температура\n"
        "1 — Тренд, сезонность и остатки\n"
        "2 — Скользящее среднее\n"
        "3 — Годовая сезонность и спектр Фурье\n"
        "print — Таблица среднесуточных температур\n"
        "exit — Выход\nВыберите команду: "
    )
    while True:
        try:
            command = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if command == "exit":
            break
        if command == "print":
            print(analysis["series"].to_string())
        elif command in actions:
            fig = actions[command](analysis)
            plt.show()
            plt.close(fig)
        else:
            print("Неизвестная команда. Выберите 0, 1, 2, 3, print или exit.")


def main():
    parser = argparse.ArgumentParser(description="Задание 2. Температуры Воронежа за 2024–2025 годы")
    parser.add_argument("--save", action="store_true", help="Сохранить графики без меню")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    weather = load_weather()
    analysis = analyse_weather(weather)
    print_results(analysis)
    for index, draw in enumerate((draw_temperature_plot, draw_decomposition_plot, draw_spectrum_plot), 1):
        fig = draw(analysis)
        fig.savefig(OUTPUT / FIGURES[index], dpi=160, bbox_inches="tight")
        plt.close(fig)
    print(f"Графики: {OUTPUT}")
    print(f"Описание и отчёт: {ROOT / 'README.md'}")
    if not args.save and sys.stdin.isatty():
        menu(analysis)


if __name__ == "__main__":
    main()
