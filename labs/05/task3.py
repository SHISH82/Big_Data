"""Полиномиальные регрессии температуры и прямой отбор степеней."""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

if "--save" in sys.argv or not sys.stdin.isatty():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

from weather_parser import load_weather


OUTPUT = Path(__file__).resolve().parent / "results"
MAX_DEGREE = 10
IMPROVEMENT_TOL = 1e-10
FIGURES = ("02_Полиномиальные_регрессии_температуры.png", "03_Прямой_отбор_степени_полинома.png")


@dataclass
class SelectionResult:
    model: object
    powers: list
    history: list
    center: float
    scale: float


def fit_model(z, temperature, powers):
    design = np.column_stack([np.ones(len(z))] + [z ** power for power in powers])
    if len(z) <= design.shape[1] or np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("Недостаточно наблюдений или матрица полинома не имеет полного ранга")
    return sm.OLS(temperature, design).fit()


def model_metrics(model):
    return {"RSS": float(model.ssr), "RSE": float(np.sqrt(model.mse_resid)),
            "R_squared": float(model.rsquared), "adjusted_R_squared": float(model.rsquared_adj),
            "AIC": float(model.aic), "BIC": float(model.bic)}


def forward_selection(years, temperature, max_degree=MAX_DEGREE):
    years, temperature = np.asarray(years, dtype=float), np.asarray(temperature, dtype=float)
    if years.ndim != 1 or years.shape != temperature.shape or len(years) <= max_degree + 1:
        raise ValueError("Число пар наблюдений должно превышать максимальное число коэффициентов")
    if not np.isfinite(years).all() or not np.isfinite(temperature).all() or np.ptp(years) == 0:
        raise ValueError("Нужны конечные данные и различные годы")
    center, scale = float(years.mean()), float(np.ptp(years) / 2)
    z = (years - center) / scale
    selected = []
    remaining = list(range(1, max_degree + 1))
    current = fit_model(z, temperature, [])
    history = [{"step": 0, "added_power": "", "powers": "", "degree": 0,
                "accepted": True, **model_metrics(current)}]
    while remaining:
        candidates = [(power, fit_model(z, temperature, selected + [power])) for power in remaining]
        power, best = min(candidates, key=lambda candidate: (candidate[1].ssr, candidate[0]))
        accepted = best.rsquared_adj > current.rsquared_adj + IMPROVEMENT_TOL
        proposed = selected + [power]
        history.append({"step": len(selected) + 1, "added_power": power,
                        "powers": ";".join(map(str, proposed)), "degree": max(proposed),
                        "accepted": bool(accepted), **model_metrics(best)})
        if not accepted:
            break
        selected, current = proposed, best
        remaining.remove(power)
    return SelectionResult(current, selected, history, center, scale)


def predict(result, years):
    z = (np.asarray(years, dtype=float) - result.center) / result.scale
    design = np.column_stack([np.ones(len(z))] + [z ** power for power in result.powers])
    return result.model.predict(design)


def save_table(filename, rows):
    with filename.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def draw_plots(years, temperature, result, full_models):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.grid": True, "grid.alpha": 0.2})
    grid = np.linspace(years.min(), years.max(), 400)
    grid_z = (grid - result.center) / result.scale
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), layout="constrained",
                              gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(years, temperature, "o-", color="gray", alpha=0.65, label="Годовые температуры")
    for degree, color in ((1, "tab:blue"), (2, "tab:green"), (3, "tab:orange")):
        design = np.column_stack([grid_z ** p for p in range(degree + 1)])
        axes[0].plot(grid, full_models[degree].predict(design), color=color, alpha=0.75,
                     label=f"Полный полином степени {degree}")
    degree = max(result.powers, default=0)
    axes[0].plot(grid, predict(result, grid), color="tab:red", linewidth=2.5,
                 label=f"Прямой отбор: степень {degree}, члены {result.powers}")
    axes[0].set(title="Среднегодовая температура Воронежа, 1986–2025",
                xlabel="Год", ylabel="Температура, °C")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[1].stem(years, result.model.resid, basefmt=" ")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set(title="Остатки модели, выбранной прямым отбором", xlabel="Год", ylabel="Температура − прогноз, °C")

    selection_fig, selection_axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    degrees = list(full_models)
    selection_axes[0].plot(degrees, [full_models[d].rsquared for d in degrees], "o-", label="R²")
    selection_axes[0].plot(degrees, [full_models[d].rsquared_adj for d in degrees], "s-", label="Скорректированный R²")
    selection_axes[0].set(title="Полные полиномы: все степени от 0 до d",
                          xlabel="Степень d", ylabel="Доля объяснённой вариации", xticks=degrees)
    selection_axes[0].legend()
    accepted = [row for row in result.history if row["accepted"]]
    selection_axes[1].plot([row["step"] for row in accepted],
                           [row["adjusted_R_squared"] for row in accepted], "o-", label="Принятые шаги")
    for row in accepted:
        if row["step"]:
            selection_axes[1].annotate(f"+ z^{row['added_power']}",
                (row["step"], row["adjusted_R_squared"]), xytext=(0, -18),
                textcoords="offset points", ha="center")
    rejected = [row for row in result.history if not row["accepted"]]
    if rejected:
        row = rejected[0]
        selection_axes[1].scatter(row["step"], row["adjusted_R_squared"], color="tab:red", marker="x", s=70,
                                  label=f"Остановка: z^{row['added_power']} отклонён")
    selection_axes[1].set(title="Прямой отбор с правилом остановки",
                          xlabel="Шаг", ylabel="Скорректированный R²",
                          xticks=[row["step"] for row in result.history])
    selection_axes[1].legend(loc="lower right")
    return fig, selection_fig


def main():
    parser = argparse.ArgumentParser(description="Лабораторная 05, задание 3: температуры Воронежа")
    parser.add_argument("--save", action="store_true", help="Сохранить графики без открытия окон")
    args = parser.parse_args()
    records = load_weather()
    years = np.array([row["year"] for row in records], dtype=float)
    temperature = np.array([row["temp"] for row in records], dtype=float)
    result = forward_selection(years, temperature)
    z = (years - result.center) / result.scale
    full_models = {d: fit_model(z, temperature, list(range(1, d + 1))) for d in range(MAX_DEGREE + 1)}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    save_table(OUTPUT / "polynomial_models.csv",
               [{"degree": d, **model_metrics(model)} for d, model in full_models.items()])
    save_table(OUTPUT / "forward_selection.csv", result.history)
    save_table(OUTPUT / "selected_coefficients.csv",
               [{"power": power, "coefficient": float(coefficient), "year_center": result.center,
                 "year_scale": result.scale} for power, coefficient in zip([0] + result.powers, result.model.params)])
    print(f"Воронеж: {len(years)} годовых значений, {int(years[0])}–{int(years[-1])}")
    print(f"z = (год − {result.center:g}) / {result.scale:g}")
    for row in result.history:
        status = "принят" if row["accepted"] else "отклонён, остановка"
        print(f"Шаг {row['step']}: +степень {str(row['added_power']):2}, "
              f"RSS={row['RSS']:.6f}, Adj R²={row['adjusted_R_squared']:.6f}, {status}")
    print(f"Выбранные степени: {result.powers}; степень полинома: {max(result.powers, default=0)}")
    for power, coefficient in zip([0] + result.powers, result.model.params):
        print(f"Коэффициент при z^{power}: {coefficient:.12g}")
    for name, value in model_metrics(result.model).items():
        print(f"{name}: {value:.10g}")
    figures = draw_plots(years, temperature, result, full_models)
    for figure, filename in zip(figures, FIGURES):
        figure.savefig(OUTPUT / filename, dpi=160, bbox_inches="tight")
    print(f"Таблицы и графики: {OUTPUT}")
    if not args.save and sys.stdin.isatty():
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
