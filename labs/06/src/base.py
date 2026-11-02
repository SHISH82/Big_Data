"""Полиномиальные Ridge и Lasso с единой нормировкой штрафа."""

import numpy as np
from sklearn.linear_model import LassoLars, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


DEGREE = 11
ACTIVE_TOL = 1e-7


def build_model(method, penalty, sample_size):
    if penalty < 0 or not np.isfinite(penalty) or sample_size < 2:
        raise ValueError("Нужны конечный неотрицательный штраф и не менее двух наблюдений")
    if method == "OLS":
        if penalty != 0:
            raise ValueError("Для МНК штраф равен нулю")
        estimator = LinearRegression()
    elif method == "Ridge" and penalty > 0:
        estimator = Ridge(alpha=penalty, solver="svd")
    elif method == "Lasso" and penalty > 0:
        estimator = LassoLars(alpha=penalty / (2 * sample_size), max_iter=1000,
                              eps=1e-14, fit_path=True)
    else:
        raise ValueError("Выберите OLS при λ=0 или Ridge/Lasso при λ>0")
    return make_pipeline(PolynomialFeatures(DEGREE, include_bias=False),
                         StandardScaler(), estimator)


def get_coefficients(model):
    scaler, estimator = model.steps[-2][1], model.steps[-1][1]
    theta = np.asarray(estimator.coef_).ravel()
    intercept = float(np.asarray(estimator.intercept_).ravel()[0])
    beta = theta / scaler.scale_
    beta0 = intercept - float(beta @ scaler.mean_)
    return np.r_[beta0, beta], np.r_[intercept, theta]


def kkt_violation(model, method, penalty, x, y):
    features = model[:-1].transform(np.asarray(x).reshape(-1, 1))
    residual = np.asarray(y) - np.asarray(model.predict(np.asarray(x).reshape(-1, 1))).ravel()
    theta = get_coefficients(model)[1][1:]
    gradient = -2 * features.T @ residual
    intercept_error = abs(2 * residual.sum())
    if method == "Ridge":
        error = np.max(np.abs(gradient + 2 * penalty * theta))
    elif method == "Lasso":
        active = np.abs(theta) > ACTIVE_TOL
        error = max(np.max(np.abs(gradient[active] + penalty * np.sign(theta[active])), initial=0),
                    np.max(np.maximum(np.abs(gradient[~active]) - penalty, 0), initial=0))
    else:
        error = np.max(np.abs(gradient))
    return float(max(intercept_error, error))


def fit_model(method, penalty, x, y):
    x, y = np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Нужны пары конечных чисел x,y")
    model = build_model(method, penalty, len(x)).fit(x.reshape(-1, 1), y)
    error = kkt_violation(model, method, penalty, x, y)
    if not np.isfinite(error) or error > 1e-7 * max(1., penalty, np.linalg.norm(y)):
        raise RuntimeError(f"Не выполнены условия оптимальности {method}: ошибка {error:.3e}")
    return model


def loocv_mse(method, penalty, x, y):
    errors = []
    for i in range(len(x)):
        mask = np.arange(len(x)) != i
        model = fit_model(method, penalty, x[mask], y[mask])
        prediction = np.asarray(model.predict(np.array([[x[i]]]))).ravel()[0]
        errors.append((prediction - y[i]) ** 2)
    return float(np.mean(errors))


def orthogonal_solutions(y, penalty):
    if penalty < 0 or not np.isfinite(penalty):
        raise ValueError("Штраф должен быть конечным и неотрицательным")
    y = np.asarray(y, dtype=float)
    return y / (1 + penalty), np.sign(y) * np.maximum(np.abs(y) - penalty / 2, 0)
