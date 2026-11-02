"""Графики влияния штрафа и шума на полиномиальные модели."""

import matplotlib.pyplot as plt
import numpy as np

from base import get_coefficients


FIGURES = ("01_Сравнение_штрафов_без_шума.png", "02_Модели_при_разных_уровнях_шума.png",
           "03_Выбор_штрафа_и_изменения_коэффициентов.png")


def draw_plots(x, samples, penalties, models, metrics, selected, ols_models):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.grid": True, "grid.alpha": 0.2})
    grid = np.linspace(x.min(), x.max(), 401)
    reference = 1 + grid ** 3
    methods = (("Ridge", "tab:blue", "-"), ("Lasso", "tab:green", "--"))

    clean_fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
    for ax, requested in zip(axes.flat, (0, 0.001, 0.01, 0.1, 1, 10)):
        if requested == 0:
            ax.plot(grid, ols_models[0.].predict(grid[:, None]), color="tab:orange", label="МНК, минимальная норма")
        else:
            penalty = float(penalties[np.argmin(np.abs(penalties - requested))])
            for method, color, style in methods:
                ax.plot(grid, models[(0., method, penalty)].predict(grid[:, None]), color=color,
                         linestyle=style, label=method)
        ax.plot(grid, reference, "k:", label="Кубический эталон 1+x³")
        ax.scatter(x, samples[0.], color="tab:red", zorder=4, s=30, label="Исходные точки")
        ax.set(title=f"λ = {requested:g}", xlabel="x", ylabel="y")
        ax.legend(fontsize=8)
    clean_fig.suptitle("Признаки x,…,x¹¹: влияние штрафа на чистых данных")

    noise_fig, noise_axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for ax, (sigma, y) in zip(noise_axes.flat, samples.items()):
        ax.plot(grid, reference, "k:", label="Кубический эталон 1+x³")
        ax.scatter(x, y, color="tab:red", zorder=4, label="Наблюдения")
        for method, color, style in methods:
            penalty = selected[(sigma, method)]["lambda"]
            model = models[(sigma, method, penalty)]
            ax.plot(grid, model.predict(grid[:, None]), color=color, linestyle=style,
                     label=f"{method}, λ={penalty:.4g}")
        ax.set(title=f"σ = {sigma:g}, штраф выбран LOOCV", xlabel="x", ylabel="y")
        ax.legend(fontsize=9)
    noise_fig.suptitle("Одна и та же реализация стандартного шума, умноженная на σ")

    diagnostic_fig, diagnostic_axes = plt.subplots(2, 2, figsize=(14, 11), layout="constrained")
    sigmas = list(samples)
    for col, (method, _, _) in enumerate(methods):
        for sigma in sigmas:
            rows = [r for r in metrics if r["sigma"] == sigma and r["method"] == method]
            diagnostic_axes[0, col].plot([r["lambda"] for r in rows], [r["loocv_mse"] for r in rows],
                                         label=f"σ={sigma:g}")
            best = selected[(sigma, method)]
            diagnostic_axes[0, col].plot(best["lambda"], best["loocv_mse"], "o")
        diagnostic_axes[0, col].set(xscale="log", yscale="log", title=f"{method}: выбор штрафа",
                                   xlabel="λ в единой нормировке RSS + λ · штраф", ylabel="MSE LOOCV")
        diagnostic_axes[0, col].legend()
    fixed = float(penalties[np.argmin(np.abs(penalties - 0.1))])
    differences = []
    for method, _, _ in methods:
        baseline = get_coefficients(models[(0., method, fixed)])[1]
        differences.append(np.column_stack([get_coefficients(models[(s, method, fixed)])[1] - baseline for s in sigmas]))
    limit = max(float(np.max(np.abs(values))) for values in differences)
    for col, ((method, _, _), values) in enumerate(zip(methods, differences)):
        ax = diagnostic_axes[1, col]
        im = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
        ax.set_xticks(range(len(sigmas)), [f"{s:g}" for s in sigmas])
        ax.set_yticks(range(12), ["Своб. член"] + [f"x^{p}" for p in range(1, 12)])
        ax.set(title=f"{method}: изменение коэффициентов при λ={fixed:g}",
               xlabel="Стандартное отклонение шума σ", ylabel="Стандартизованные коэффициенты")
        diagnostic_fig.colorbar(im, ax=ax, label="θ(σ) − θ(0)")
    return clean_fig, noise_fig, diagnostic_fig
