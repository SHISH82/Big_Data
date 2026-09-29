"""Выделение тренда и проверка остатков. Запуск: python3 labs/02_01/main.py."""

import csv
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
from scipy.stats import kendalltau


H = 0.05
SEED = 18022004
WINDOWS = (21, 51, 111)
ALPHA = 0.05
RESULTS_DIR = Path(__file__).resolve().parent / "results"
PLOT_FILES = {
    "series": "01_Модельный_ряд_и_точный_тренд.png",
    "all_trends": "02_Сравнение_среднего_и_медианы_с_точным_трендом.png",
    "trend": "03_Оценки_тренда_по_методам_и_окнам.png",
    "residuals": "04_Остатки_после_вычитания_трендов.png",
}


def generate_series(seed=SEED):
    k = np.arange(501)
    exact_trend = np.sqrt(k * H)
    # Локальный генератор даёт тот же ряд, что np.random.seed(seed) в примере.
    noise = np.random.RandomState(seed).normal(0, 1, len(k))
    return k, exact_trend, exact_trend + noise


def moving_trend(series, window, method):
    """Окно [k-m, k+m]; на краях берутся доступные элементы, как в примере."""
    series = np.asarray(series, dtype=float)
    if window < 3 or window % 2 == 0 or window > len(series):
        raise ValueError("Окно должно быть нечётным, от 3 до длины ряда.")
    if method not in ("mean", "median"):
        raise ValueError("Метод должен быть mean или median.")

    m = window // 2
    trend = np.empty(len(series))
    for k in range(len(series)):
        start = max(0, k - m)
        end = min(len(series), k + m + 1)
        values = series[start:end]
        trend[k] = np.mean(values) if method == "mean" else np.median(values)
    return trend


def count_turning_points(series):
    """Учитываются только строгие локальные максимумы и минимумы."""
    differences = np.diff(series)
    return int(np.count_nonzero(differences[:-1] * differences[1:] < 0))


def normal_p_value(z):
    return math.erfc(abs(z) / math.sqrt(2))


def turning_points_test(series, permutations=4999, seed=2026):
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < 3 or not np.all(np.isfinite(series)):
        raise ValueError("Нужно минимум 3 конечных значения.")
    turning = count_turning_points(series)
    expected = 2 * (n - 2) / 3
    variance = (16 * n - 29) / 90 if n >= 4 else 2 / 9
    z = (turning - expected) / math.sqrt(variance)
    p_normal = normal_p_value(z)
    p_value = p_normal
    has_ties = len(np.unique(series)) < n

    # Формулы лекции предполагают непрерывное распределение без совпадений.
    # При совпадениях проверяем случайный порядок перестановками тех же чисел.
    if has_ties:
        rng = np.random.default_rng(seed)
        simulated = np.array([
            count_turning_points(rng.permutation(series))
            for _ in range(permutations)
        ])
        lower = (1 + np.count_nonzero(simulated <= turning)) / (permutations + 1)
        upper = (1 + np.count_nonzero(simulated >= turning)) / (permutations + 1)
        p_value = min(1.0, 2 * min(lower, upper))

    return {
        "turning_points": turning, "turning_expected": expected,
        "turning_z": z, "turning_p_normal": p_normal,
        "turning_p": p_value,
        "turning_method": "permutation" if has_ties else "normal",
    }


def kendall_test(series):
    """Связь значения с номером отсчёта: scipy.stats.kendalltau, как в примере."""
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < 3 or not np.all(np.isfinite(series)):
        raise ValueError("Нужно минимум 3 конечных значения.")
    increasing = 0
    decreasing = 0
    for i in range(n - 1):
        increasing += int(np.count_nonzero(series[i + 1:] > series[i]))
        decreasing += int(np.count_nonzero(series[i + 1:] < series[i]))

    pairs = n * (n - 1) // 2
    ties = pairs - increasing - decreasing
    score = increasing - decreasing
    # Без совпадений это в точности 4*P/(n*(n-1)) - 1 из лекции.
    tau_a = score / pairs
    # tau-b учитывает равные значения, возникающие после медианного сглаживания.
    test = kendalltau(np.arange(n), series, method="asymptotic")
    _, counts = np.unique(series, return_counts=True)
    tie_correction = sum(int(t) * (int(t) - 1) * (2 * int(t) + 5) for t in counts)
    variance_score = (n * (n - 1) * (2 * n + 5) - tie_correction) / 18
    z = score / math.sqrt(variance_score) if variance_score > 0 else 0.0
    return {
        "kendall_P": increasing, "kendall_Q": decreasing, "tied_pairs": ties,
        "kendall_tau_a": tau_a, "kendall_tau": float(test.statistic),
        "kendall_z": z, "kendall_p": float(test.pvalue),
    }


def analyze_series(series, exact_trend):
    results = []
    for method, label in (("mean", "Среднее"), ("median", "Медиана")):
        for window in WINDOWS:
            trend = moving_trend(series, window, method)
            residuals = series - trend
            error = trend - exact_trend
            metrics = {
                "method": label, "window": window, "n": len(series),
                "mse": float(np.mean(error ** 2)),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error ** 2))),
                **turning_points_test(residuals),
                **kendall_test(residuals),
            }
            results.append({"metrics": metrics, "trend": trend, "residuals": residuals})
    return results


def save_plots(k, series, exact_trend, results, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4), layout="constrained")
    ax.plot(k, series, color="0.65", linewidth=0.8, label="Модельный ряд")
    ax.plot(k, exact_trend, color="black", linewidth=2, label="Точный тренд √(k·h)")
    ax.set(xlabel="k", ylabel="Значение", title=f"Модельный ряд, h={H}, seed={SEED}")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.savefig(directory / PLOT_FILES["series"], dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    ax.plot(k, series, color="0.8", linewidth=0.6, label="Модельный ряд")
    ax.plot(k, exact_trend, color="black", linewidth=2, label="Точный тренд")
    colors = {21: "tab:blue", 51: "tab:orange", 111: "tab:green"}
    for result in results:
        row = result['metrics']
        style = "--" if row['method'] == "Среднее" else "-"
        ax.plot(k, result['trend'], linestyle=style, color=colors[row['window']],
                linewidth=1.2, label=f"{row['method']}, окно {row['window']}")
    ax.set(xlabel="k", ylabel="Значение", title="Сравнение всех шести оценок тренда")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.savefig(directory / PLOT_FILES["all_trends"], dpi=160)
    plt.close(fig)

    for kind in ("trend", "residuals"):
        fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True, sharey=True,
                                 layout="constrained")
        for index, result in enumerate(results):
            ax = axes[index % 3, index // 3]
            metrics = result["metrics"]
            if kind == "trend":
                ax.plot(k, series, color="0.85", linewidth=0.6, label="Ряд")
                ax.plot(k, exact_trend, "k--", linewidth=1.5, label="Точный тренд")
                ax.plot(k, result[kind], color="tab:blue", label="Оценка тренда")
            else:
                ax.plot(k, result[kind], color="tab:blue", linewidth=0.7)
                ax.axhline(0, color="black", linewidth=1)
            ax.set_title(f"{metrics['method']}, окно {metrics['window']}")
            ax.set(xlabel="k", ylabel="Тренд" if kind == "trend" else "Остаток")
            ax.grid(alpha=0.25)
        if kind == "trend":
            axes[0, 0].legend(fontsize=8)
        fig.savefig(directory / PLOT_FILES[kind], dpi=160)
        plt.close(fig)


def save_report(results, directory):
    critical = NormalDist().inv_cdf(1 - ALPHA / 2)
    rows = [item["metrics"] for item in results]
    best = min(rows, key=lambda row: row["rmse"])
    lines = [
        "# Лабораторная работа 02_01 — результаты", "",
        "## Запуск", "",
        "Из корня проекта: `python3 -m pip install -r labs/02_01/requirements.txt`, "
        "затем `python3 labs/02_01/main.py`. Файлы результатов сохраняются рядом с программой "
        "в `results/`. Повторный запуск обновляет их.", "",
        "## Модель и обработка краёв", "",
        f"xₖ = √(k·{H}) + εₖ, k = 0,…,500; εₖ — независимые N(0,1). "
        f"501 отсчёт, генератор NumPy RandomState, seed={SEED}, как в принятом примере.", "",
        "Для окна w=2m+1 среднее равно сумме x[k−m:k+m+1], делённой на w; "
        "медиана — центральному элементу отсортированного окна. "
        "На краях окно сокращается: start=max(0,k−m), end=min(501,k+m+1); "
        "используются элементы x[start:end]. Среднее делится на фактическое число элементов. "
        "Так же обработаны края в принятом примере. Все шесть оценок содержат 501 значение.", "",
        f"![Модельный ряд и точный тренд]({PLOT_FILES['series']})", "",
        "## Сравнение с точным трендом", "",
        "Все ошибки рассчитаны на полном ряде k=0,…,500, включая края, как в примере. "
        "MSE = mean((оценка − точный тренд)²); RMSE = √MSE; "
        "MAE = mean(|оценка − точный тренд|).", "",
        "| Метод | Окно | MSE | MAE | RMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['method']} | {row['window']} | {row['mse']:.6f} | {row['mae']:.6f} | {row['rmse']:.6f} |")
    lines += ["", f"![Сравнение всех шести трендов]({PLOT_FILES['all_trends']})", "",
              f"![Оценки по методам и окнам]({PLOT_FILES['trend']})", "",
              "## Проверка остатков", "",
              "Для каждого из шести трендов вычислены остатки rₖ=xₖ−оценка тренда. "
              "Оба критерия применяются ко всем 501 остаткам каждого варианта. "
              f"Двусторонние проверки: α={ALPHA}, критическое |z|≈{critical:.4f}. "
              "При p<α гипотеза отвергается. Поправка на множественные проверки не применяется.", "",
              "### Поворотные точки", "",
              "T — число строгих локальных экстремумов. По слайду 6: "
              "E(T)=2(n−2)/3, D(T)=(16n−29)/90; z=(T−E(T))/√D(T). "
              "p норм. — двусторонняя вероятность стандартного нормального распределения.", "",
              "Если значения совпадают, предпосылка непрерывного распределения нарушена. "
              "В таком случае решение принимается по двустороннему перестановочному p: "
              "4999 случайных перестановок остатков с сохранением совпадений, seed=2026; "
              "p=min(1, 2·min((1+#(T*≤T))/5000, (1+#(T*≥T))/5000)). "
              "Это проверка случайного порядка условно на наблюдаемых значениях. "
              "z и p норм. приведены для сопоставления с лекцией.", "",
              "| Метод | Окно | n | T | E(T) | z | p норм. | p для решения | Способ |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append(f"| {row['method']} | {row['window']} | {row['n']} | {row['turning_points']} | "
                     f"{row['turning_expected']:.3f} | {row['turning_z']:.4f} | "
                     f"{row['turning_p_normal']:.4f} | {row['turning_p']:.4f} | {row['turning_method']} |")
    lines += ["", "### Коэффициент Кендела", "",
              "Сравниваются все пары i<j. P — число rⱼ>rᵢ, Q — число rⱼ<rᵢ, "
              "S=P−Q. Без совпадений формула слайда 7: "
              "τ=4P/[n(n−1)]−1, D(τ)=2(2n+5)/[9n(n−1)]. "
              "В расчёте используется scipy.stats.kendalltau, как в принятом примере. "
              "При совпадениях он возвращает tau-b: τ=S/√(N·(N−B)), "
              "где N=n(n−1)/2, B — число равных пар. Без совпадений tau-a и tau-b совпадают.", "",
              "При совпадениях используется поправка: "
              "D(S)=[n(n−1)(2n+5)−Σ t(t−1)(2t+5)]/18, "
              "где t — число одинаковых значений в каждой группе; z=S/√D(S). "
              "Индексы времени уникальны. Нулевые остатки медианного сглаживания "
              "не заменяются искусственным шумом. Нормальная проверка приближённая. "
              "Формула поправки сверена с [реализацией SciPy](https://github.com/scipy/scipy/blob/v1.17.1/scipy/stats/_stats_py.py).", "",
              "| Метод | Окно | P | Q | Равные пары | τ (tau-b) | z | p |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['method']} | {row['window']} | {row['kendall_P']} | "
                     f"{row['kendall_Q']} | {row['tied_pairs']} | {row['kendall_tau']:.6f} | "
                     f"{row['kendall_z']:.4f} | {row['kendall_p']:.4f} |")
    lines += ["", f"![Остатки после вычитания трендов]({PLOT_FILES['residuals']})", "",
              "## Выводы", "",
              f"На этой реализации минимальная RMSE у метода «{best['method']}», "
              f"окно {best['window']}: {best['rmse']:.6f}. "
              "Это результат конкретного случайного ряда, а не универсальный выбор окна.", ""]
    for label in ("Среднее", "Медиана"):
        selected = [row for row in rows if row['method'] == label]
        values = " → ".join(f"{row['rmse']:.6f}" for row in selected)
        lines.append(f"- {label}: RMSE при окнах 21 → 51 → 111: {values}.")
    lines += ["", "Большое окно сильнее подавляет шум, но может сглаживать изгибы тренда. "
              "Медиана устойчива к выбросам; здесь шум нормальный и дополнительные выбросы "
              "не вводились, поэтому преимущества медианы заранее не предполагаются.", ""]
    for row in rows:
        turning = "отвергает случайный порядок" if row['turning_p'] < ALPHA else "не отвергает случайный порядок"
        kendall = ("обнаруживает " + ("возрастающий" if row['kendall_tau'] > 0 else "убывающий") + " тренд"
                   if row['kendall_p'] < ALPHA else "не обнаруживает значимого монотонного тренда")
        lines.append(f"- {row['method']}, окно {row['window']}: критерий поворотных точек {turning}; "
                     f"критерий Кендела {kendall}.")
    lines += ["", "Неотвержение гипотезы не доказывает случайность или независимость. "
              "Два критерия выявляют разные нарушения: локальные колебания и монотонный тренд. "
              "Соседние оценки используют общие наблюдения, поэтому после вычитания тренда "
              "остатки могут быть зависимыми даже при независимом исходном шуме. "
              "Здесь критерии применяются как диагностика остатков по заданию.", "",
              "Исходные данные, оценки и остатки: [series.csv](series.csv). "
              "Численные результаты: [metrics.csv](metrics.csv). "
              "Основа решения — «Лекция 9», слайды 6, 7, 9, 13, 14.", "",
              "## Сверка с принятым примером", "",
              "[Код примера](https://github.com/azya0/big_data_2025/blob/main/labs/02_lost/main.py), "
              "[зависимости](https://github.com/azya0/big_data_2025/blob/main/labs/02_lost/requirements.txt), "
              "[условие](https://github.com/azya0/big_data_2025/blob/main/labs/02_lost/2_lost.png).", "",
              "| Пункт | Реализация |",
              "|---|---|",
              "| 1. Модельный ряд | Те же 501 отсчёт, h=0.05, нормальный шум и seed=18022004. |",
              "| 2. Скользящее среднее | Окна 21, 51, 111; на краях — доступная часть окна. |",
              "| 3. Скользящая медиана | Те же окна и обработка краёв. На графиках показана именно медиана. |",
              "| 4. Сравнение с точным трендом | Графики, MSE на всех 501 точках, численные выводы. |",
              "| 5. Проверка остатков | Строгие поворотные точки, E(T), коэффициент scipy.stats.kendalltau для всех шести вариантов. |", "",
              "Дополнительно к числам из примера приведены p-value и решения при α=0.05. "
              "Для совпадающих остатков медианы проверка поворотных точек уточнена перестановками; "
              "формулы лекции также приведены. Графики сохраняются в PNG. "
              "В requirements.txt перечислены используемые библиотеки: NumPy, Matplotlib и SciPy; "
              "остальные пакеты из окружения примера для этой работы не требуются.", ""]
    (directory / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    k, exact_trend, series = generate_series()
    results = analyze_series(series, exact_trend)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    columns = [k, k * H, exact_trend, series, series - exact_trend]
    names = ["k", "time", "exact_trend", "series", "noise"]
    for index, result in enumerate(results):
        method = "mean" if index < 3 else "median"
        suffix = f"{method}_{result['metrics']['window']}"
        columns.extend([result['trend'], result['residuals']])
        names.extend([f"trend_{suffix}", f"residual_{suffix}"])
    np.savetxt(RESULTS_DIR / "series.csv", np.column_stack(columns), delimiter=",",
               header=",".join(names), comments="", fmt="%.12g")
    with (RESULTS_DIR / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0]['metrics'].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(item['metrics'] for item in results)
    save_plots(k, series, exact_trend, results, RESULTS_DIR)
    save_report(results, RESULTS_DIR)

    print(f"501 отсчёт, seed={SEED}, уровень значимости {ALPHA}")
    print("Метод     Окно   n     MSE     RMSE     T    E(T)       tau   p(повороты)  p(Кендел)")
    for result in results:
        row = result['metrics']
        print(f"{row['method']:8} {row['window']:4} {row['n']:4} "
              f"{row['mse']:8.4f} {row['rmse']:8.4f} "
              f"{row['turning_points']:4} {row['turning_expected']:7.2f} {row['kendall_tau']:9.5f} "
              f"{row['turning_p']:12.4f} {row['kendall_p']:10.4f}")
        turning = "отвергается" if row['turning_p'] < ALPHA else "не отвергается"
        kendall = "обнаружен" if row['kendall_p'] < ALPHA else "не обнаружен"
        print(f"  Случайный порядок по поворотным точкам: {turning}; монотонный тренд по Кенделу: {kendall}.")
    best = min((result['metrics'] for result in results), key=lambda row: row['mse'])
    print(f"Минимальная MSE: {best['method']}, окно {best['window']}, MSE={best['mse']:.6f}.")
    print(f"Графики, таблицы и выводы: {RESULTS_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
