"""RSS, RSE, R² и диагностика множественной регрессии."""

import argparse
import csv
import sys

import matplotlib

if "--save" in sys.argv or not sys.stdin.isatty():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from task1 import OUTPUT, SAMPLE_SIZE, SEED, TRUE_COEFFICIENTS, create_model, save_sample


FIGURE = "01_Множественная_регрессия_и_остатки.png"


def get_model_statistics(model):
    rss = float(np.sum(model.resid ** 2))
    return {"RSS": rss, "RSE": float(np.sqrt(rss / model.df_resid)),
            "R_squared": float(model.rsquared), "adjusted_R_squared": float(model.rsquared_adj),
            "F": float(model.fvalue), "F_pvalue": float(model.f_pvalue)}


def draw_plot(model, metrics):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.grid": True, "grid.alpha": 0.2})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    y, fitted = model.model.endog, model.fittedvalues
    low, high = min(y.min(), fitted.min()), max(y.max(), fitted.max())
    axes[0].scatter(fitted, y)
    axes[0].plot([low, high], [low, high], "--", color="tab:red", label="Совпадение y = ŷ")
    axes[0].set(title="Наблюдаемые и расчётные отклики", xlabel="Расчётный отклик ŷ", ylabel="Наблюдаемый отклик y")
    axes[0].legend(loc="lower right")
    axes[0].text(0.04, 0.96,
                 f"n = {int(model.nobs)}, df = {int(model.df_resid)}\nRSS = {metrics['RSS']:.4f}\n"
                 f"RSE = {metrics['RSE']:.4f}\nR² = {metrics['R_squared']:.6f}",
                 transform=axes[0].transAxes, va="top",
                 bbox={"facecolor": "white", "edgecolor": "gray", "alpha": 0.9})
    axes[1].scatter(fitted, model.resid)
    axes[1].axhline(0, color="black", linestyle="--")
    axes[1].set(title="Остатки множественной регрессии", xlabel="Расчётный отклик ŷ", ylabel="Остаток y − ŷ")
    fig.suptitle("Модель y = 1 + 3x₁ − 2x₂ + x₃ + e, e ~ N(0,1)")
    return fig


def main():
    parser = argparse.ArgumentParser(description="Лабораторная 05, задания 1–2")
    parser.add_argument("--save", action="store_true", help="Сохранить график без открытия окна")
    args = parser.parse_args()
    model, size = create_model()
    metrics = get_model_statistics(model)
    print(f"Множественная регрессия: n={size}, seed={SEED}")
    for name, truth, estimate in zip(("const", "x1", "x2", "x3"), TRUE_COEFFICIENTS, model.params):
        print(f"{name:5}: точное = {truth: .0f}, оценка = {estimate: .9f}")
    for name, value in metrics.items():
        print(f"{name}: {value:.10g}")
    print(f"Число степеней свободы: {size} − 4 = {int(model.df_resid)}")
    print(f"Объяснено {100 * metrics['R_squared']:.2f}% вариации отклика в сгенерированной выборке.")
    print("Высокий R² описывает подгонку; он не заменяет анализ остатков и проверку на новых данных.")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    save_sample(model, OUTPUT / "model_sample.csv")
    with (OUTPUT / "multiple_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("n", "seed", "beta0", "beta1", "beta2", "beta3", *metrics))
        writer.writerow((SAMPLE_SIZE, SEED, *model.params, *metrics.values()))
    figure = draw_plot(model, metrics)
    figure.savefig(OUTPUT / FIGURE, dpi=160, bbox_inches="tight")
    print(f"Таблица и график: {OUTPUT}")
    if not args.save and sys.stdin.isatty():
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
