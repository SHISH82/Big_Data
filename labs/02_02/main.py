"""EMA, спектр Фурье и анализ остатков. Запуск: python labs/02_02/main.py."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

H = 0.1
SEED = 0
ALPHAS = (0.01, 0.05, 0.1, 0.3)
SIGNIFICANCE = 0.05
RESULTS_DIR = Path(__file__).resolve().parent / "results"
PLOT_FILES = {
    "trends": "01_Сравнение_EMA_с_точным_трендом.png",
    "errors": "02_Ошибки_оценки_тренда.png",
    "spectrum": "03_Амплитудный_спектр_Фурье.png",
    "residuals": "04_Анализ_остатков.png",
}


# 1. Модельный ряд и точный тренд.
def generate_series(K=500, h=H, seed=SEED):
    k = np.arange(K + 1)
    exact = 0.5 * np.sin(k * h)
    noise = np.random.RandomState(seed).normal(0, 1, len(k))
    return k, exact, exact + noise


# 2. EMA: y[0] = x[0], y[k] = alpha*x[k] + (1-alpha)*y[k-1].
def compute_ema(series, alphas=ALPHAS):
    return {
        alpha: pd.Series(series).ewm(alpha=alpha, adjust=False).mean().to_numpy()
        for alpha in alphas
    }


# 3. Сравнение с точным трендом, а не с шумным рядом.
def compare_trends(exact, trends):
    results = {}
    for alpha, trend in trends.items():
        error = trend - exact
        mse = float(np.mean(error ** 2))
        results[alpha] = {
            "alpha": alpha, "mse": mse, "rmse": np.sqrt(mse),
            "mae": float(np.mean(np.abs(error))),
            "corr": float(np.corrcoef(exact, trend)[0, 1]),
            "trend_kendall": float(stats.kendalltau(exact, trend).statistic),
        }
    return results


# 4. FFT исходного модельного ряда. Нулевая частота не участвует в поиске пика.
def fourier_spectrum(series, h=H):
    n = len(series)
    if n < 3 or h <= 0:
        raise ValueError("Нужны минимум 3 отсчёта и положительный шаг h.")
    frequencies = np.fft.rfftfreq(n, d=h)
    amplitude = np.abs(np.fft.rfft(series)) / n
    peak_index = int(np.argmax(amplitude[1:]) + 1)

    # В примере |FFT|/n. Для односторонней амплитуды удваиваются парные частоты.
    one_sided = amplitude.copy()
    one_sided[1:-1 if n % 2 == 0 else None] *= 2
    return {
        "frequencies": frequencies, "amplitude_reference": amplitude,
        "amplitude": one_sided, "peak_index": peak_index,
        "main_frequency": float(frequencies[peak_index]),
        "frequency_step": 1 / (n * h),
    }


# 5. Критерии для остатков r[k] = x[k] - y[k].
def count_turning_points(series):
    differences = np.diff(series)
    return int(np.count_nonzero(differences[:-1] * differences[1:] < 0))


def runs_test(series):
    """Критерий серий относительно медианы, как в принятом примере."""
    signs = np.asarray(series) > np.median(series)
    n1 = int(signs.sum())
    n2 = len(signs) - n1
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    runs = 1 + np.count_nonzero(signs[1:] != signs[:-1])
    expected = 1 + 2 * n1 * n2 / (n1 + n2)
    variance = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)
                / ((n1 + n2) ** 2 * (n1 + n2 - 1)))
    z = (runs - expected) / np.sqrt(variance)
    return float(z), float(2 * stats.norm.sf(abs(z)))


def test_residuals(residuals):
    n = len(residuals)
    if n <= 10:
        raise ValueError("Для проверки Льюнга — Бокса на 10 лагах нужно больше 10 точек.")
    mean = float(np.mean(residuals))
    std = float(np.std(residuals, ddof=1))
    t_test = stats.ttest_1samp(residuals, 0)
    shapiro = stats.shapiro(residuals)
    ljung = acorr_ljungbox(residuals, lags=[10], model_df=0, return_df=True)
    runs_z, runs_p = runs_test(residuals)
    turning = count_turning_points(residuals)
    turning_expected = 2 * (n - 2) / 3
    turning_variance = (16 * n - 29) / 90
    turning_z = (turning - turning_expected) / np.sqrt(turning_variance)
    kendall = stats.kendalltau(np.arange(n), residuals)
    margin = stats.t.ppf(1 - SIGNIFICANCE / 2, df=n - 1) * std / np.sqrt(n)
    return {
        "n": n, "mean": mean, "std": std,
        "mean_ci_low": mean - margin, "mean_ci_high": mean + margin,
        "t_stat": float(t_test.statistic), "p_t": float(t_test.pvalue),
        "shapiro_W": float(shapiro.statistic), "p_sh": float(shapiro.pvalue),
        "ljung_Q": float(ljung['lb_stat'].iloc[0]),
        "ljung_p": float(ljung['lb_pvalue'].iloc[0]),
        "runs_z": runs_z, "runs_p": runs_p,
        "turning_points": turning, "turning_expected": turning_expected,
        "turning_z": float(turning_z),
        "turning_p": float(2 * stats.norm.sf(abs(turning_z))),
        "residual_kendall": float(kendall.statistic),
        "kendall_p": float(kendall.pvalue),
    }


def save_plots(k, series, exact, trends, spectrum, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for kind in ("trends", "errors"):
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True,
                                 layout="constrained")
        for ax, (alpha, trend) in zip(axes.flat, trends.items()):
            if kind == "trends":
                ax.plot(k, series, color="0.8", linewidth=0.6, label="Модельный ряд")
                ax.plot(k, exact, color="black", linewidth=1.7, label="Точная синусоида")
                ax.plot(k, trend, color="tab:blue", linewidth=1.3, label="EMA")
                ax.legend(fontsize=8)
            else:
                ax.plot(k, trend - exact, color="tab:purple", linewidth=0.9)
                ax.axhline(0, color="black", linewidth=1)
            ax.set(title=f"α = {alpha}", xlabel="k",
                   ylabel="Значение" if kind == "trends" else "EMA − точный тренд")
            ax.grid(alpha=0.25)
        fig.savefig(directory / PLOT_FILES[kind], dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), layout="constrained")
    for ax in axes:
        ax.plot(spectrum['frequencies'], spectrum['amplitude'], linewidth=0.9)
        ax.axvline(1 / (2 * np.pi), color="black", linestyle="--", label="Точная f = 1/(2π)")
        ax.axvline(spectrum['main_frequency'], color="tab:red", linestyle=":",
                   label=f"Пик f = {spectrum['main_frequency']:.6f}")
        ax.set(xlabel="Частота, циклов на единицу t = k·h", ylabel="Односторонняя амплитуда")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    axes[0].set_title("Спектр модельного ряда")
    axes[1].set(title="Низкочастотная область", xlim=(0, 0.5))
    fig.savefig(directory / PLOT_FILES['spectrum'], dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(4, 3, figsize=(15, 12), layout="constrained")
    for row, (alpha, trend) in enumerate(trends.items()):
        residuals = series - trend
        axes[row, 0].plot(k, residuals, linewidth=0.7)
        axes[row, 0].axhline(0, color="black", linewidth=1)
        axes[row, 0].set(title=f"Остатки, α={alpha}", xlabel="k", ylabel="x − EMA")
        ax = axes[row, 1]
        ax.hist(residuals, bins=25, density=True, alpha=0.65)
        grid = np.linspace(residuals.min(), residuals.max(), 200)
        ax.plot(grid, stats.norm.pdf(grid, residuals.mean(), residuals.std(ddof=1)),
                color="black", label="Нормальная плотность с оценёнными μ, σ")
        ax.set(title=f"Гистограмма, α={alpha}", xlabel="Остаток", ylabel="Плотность")
        stats.probplot(residuals, dist="norm", plot=axes[row, 2])
        axes[row, 2].set(title=f"Q–Q, α={alpha}", xlabel="Теоретический квантиль N(0,1)",
                         ylabel="Квантиль остатков")
        for ax in axes[row]:
            ax.grid(alpha=0.2)
    axes[0, 1].legend(fontsize=6)
    fig.savefig(directory / PLOT_FILES['residuals'], dpi=150)
    plt.close(fig)


def save_report(rows, spectrum, directory):
    """Краткие результаты и выводы по заданию."""
    best = min(rows, key=lambda row: row['mse'])
    frequency = spectrum['main_frequency']
    lines = [
        "# Результаты лабораторной 02_02", "",
        "Запуск из корня проекта: `.venv/bin/python labs/02_02/main.py`.", "",
        f"501 отсчёт, h={H}, seed={SEED}. EMA: y₀=x₀, yₖ=αxₖ+(1−α)yₖ₋₁. "
        "Все расчёты включают начальный участок.", "",
        "## Графики для показа", "",
        f"1. [Сравнение EMA с точным трендом]({PLOT_FILES['trends']}) — "
        "соответствует `trend_comparison.png` из примера: ряд, точная синусоида и четыре EMA.",
        f"2. [Ошибки оценки тренда]({PLOT_FILES['errors']}) — "
        "соответствует `trend_comparison_errors.png`: разность EMA и точной синусоиды.",
        f"3. [Амплитудный спектр Фурье]({PLOT_FILES['spectrum']}) — для пункта о главной частоте.",
        f"4. [Анализ остатков]({PLOT_FILES['residuals']}) — остатки x−EMA, гистограммы и Q–Q графики.", "",
        "## Сравнение с точным трендом", "",
        "MSE = mean((EMA − точный тренд)²), по всем 501 точкам.", "",
        "| α | MSE | Корреляция Пирсона | Корреляция Кендела |",
        "|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['alpha']} | {row['mse']:.6f} | {row['corr']:.6f} | {row['trend_kendall']:.6f} |")
    lines += ["", f"Лучшая оценка для этой реализации шума: **α={best['alpha']}**, "
              f"MSE={best['mse']:.6f}. Малое α сильнее сглаживает, но запаздывает и дольше "
              "сохраняет влияние начального значения. Большое α пропускает больше шума.", "",
              "## Главная частота", "",
              f"**f={frequency:.8f}** циклов/ед. времени; ω={2*np.pi*frequency:.8f} рад/ед. времени; "
              f"T={1/frequency:.6f}. Теоретически f₀=1/(2π)≈0.15915494, ω₀=1. "
              f"Шаг сетки FFT: {spectrum['frequency_step']:.8f}.", "",
              "FFT рассчитано для исходного ряда. Пик ищется среди положительных частот. "
              "На графике — односторонняя амплитуда 2·|FFT|/n (кроме нулевой частоты); "
              "в примере |FFT|/n. Для этого ряда положение пика одинаковое. "
              "Небольшое отличие от теоретической частоты объясняется сеткой FFT и шумом.", "",
              "## Проверка остатков x−EMA", "",
              f"Уровень значимости {SIGNIFICANCE}. При p<0.05 гипотеза отвергается. "
              "t-тест проверяет нулевое среднее, Шапиро — Уилк — нормальность, "
              "Льюнг — Бокс — отсутствие автокорреляции до лага 10, критерий серий — "
              "порядок значений относительно медианы, Кендел — отсутствие монотонного тренда. "
              "Поворотные точки — строгие локальные экстремумы остатков.", "",
              "| α | Среднее | s | p t-теста | p Шапиро | p Льюнга — Бокса | p серий |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['alpha']} | {row['mean']:.6f} | {row['std']:.6f} | "
                     + " | ".join(f"{row[key]:.6g}" for key in ('p_t','p_sh','ljung_p','runs_p')) + " |")
    lines += ["", "| α | Поворотные точки | E(T) | p поворотов | τ остатков/времени | p Кендела |",
              "|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['alpha']} | {row['turning_points']} | {row['turning_expected']:.3f} | "
                     f"{row['turning_p']:.6g} | {row['residual_kendall']:.6f} | {row['kendall_p']:.6g} |")
    lines += ["", "## Выводы по остаткам", ""]
    criteria = {'ljung_p': 'Льюнга — Бокса', 'runs_p': 'серий',
                'turning_p': 'поворотных точек', 'kendall_p': 'Кендела'}
    for row in rows:
        rejected = [name for key, name in criteria.items() if row[key] < SIGNIFICANCE]
        random = ("нарушения случайности выявлены критериями " + ", ".join(rejected)
                  if rejected else "применённые критерии не выявили нарушений случайности")
        mean = "отвергается" if row['p_t'] < SIGNIFICANCE else "не отвергается"
        normal = "отвергается" if row['p_sh'] < SIGNIFICANCE else "не отвергается"
        lines.append(f"- α={row['alpha']}: {random}; нулевое среднее {mean}; нормальность {normal}.")
    lines += ["", "Неотвержение гипотез не доказывает независимость или нормальность. "
              "При обнаруженной автокорреляции проверки среднего и нормальности рассматриваются "
              "как приближённая диагностика. Поправка на множественные проверки не применяется.", ""]
    (directory / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    k, exact, series = generate_series()
    trends = compute_ema(series)
    comparison = compare_trends(exact, trends)
    spectrum = fourier_spectrum(series)
    rows = [{**comparison[alpha], **test_residuals(series - trend)}
            for alpha, trend in trends.items()]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    save_plots(k, series, exact, trends, spectrum, RESULTS_DIR)
    save_report(rows, spectrum, RESULTS_DIR)
    print(f"501 отсчёт; h={H}; seed={SEED}; уровень значимости={SIGNIFICANCE}")
    print("alpha    MSE      mean     p(t)    p(Shapiro)  p(Ljung)  p(runs)  p(turn)  p(Kendall)")
    for row in rows:
        print(f"{row['alpha']:5.2f} {row['mse']:9.5f} {row['mean']:9.5f} "
              + " ".join(f"{row[key]:10.5g}" for key in
                         ('p_t', 'p_sh', 'ljung_p', 'runs_p', 'turning_p', 'kendall_p')))
    best = min(rows, key=lambda row: row['mse'])
    print(f"Минимальная MSE: alpha={best['alpha']}, MSE={best['mse']:.6f}")
    frequency = spectrum['main_frequency']
    print(f"Главная частота: f={frequency:.6f} циклов/ед. времени; "
          f"omega={2 * np.pi * frequency:.6f} рад/ед. времени; период={1 / frequency:.6f}")
    print(f"Отчёт, графики и выводы: {RESULTS_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
