import numpy as np
import pandas as pd

from scipy.stats import cramervonmises_2samp


# Lorenz Curve
def lorenz_curve(y):
    y = np.asarray(y, dtype=float).reshape(-1)
    y = y[~np.isnan(y)]
    if len(y) == 0:
        return np.array([])
    y_sorted = np.sort(y)
    cum = np.cumsum(y_sorted)
    return cum / cum[-1]   # normalized to [0,1]


# Concordance Curve
def concordance_curve(y, yhat):
    y = np.asarray(y, dtype=float).reshape(-1)
    yhat = np.asarray(yhat, dtype=float).reshape(-1)
    mask = ~np.isnan(y) & ~np.isnan(yhat)
    y = y[mask]
    yhat = yhat[mask]

    if len(y) == 0:
        return np.array([])

    ord_idx = np.argsort(yhat)
    cum = np.cumsum(y[ord_idx])
    return cum / cum[-1]


# Gini
def gini_via_lorenz(y):
    y = np.asarray(y, dtype=float).reshape(-1)
    l = lorenz_curve(y)
    n = len(l)
    if n == 0:
        return np.nan
    u = np.linspace(1/n, 1, n)
    return 2 * np.mean(np.abs(u - l))


# Weighted Cramer (L1) distance between Lorenz and Concordance
def cvm1_concordance_weighted(y, yhat):
    y = np.asarray(y, dtype=float).reshape(-1)
    yhat = np.asarray(yhat, dtype=float).reshape(-1)
    mask = ~np.isnan(y) & ~np.isnan(yhat)
    y = y[mask]
    yhat = yhat[mask]

    n = len(y)
    if n == 0:
        return np.nan

    # Lorenz curve values
    ord_y = np.argsort(y)
    l = np.cumsum(y[ord_y]) / np.sum(y)

    # Concordance curve
    ord_yhat = np.argsort(yhat)
    c = np.cumsum(y[ord_yhat]) / np.sum(y)

    # weights
    weights = y[ord_y] / np.sum(y)

    return np.sum(np.abs(c - l) * weights)


# WRGA = 1 – CvM(y, yhat)/G(y)
def wrga_cramer(y, yhat):
    g = gini_via_lorenz(y)
    if not np.isfinite(g) or g == 0:
        return np.nan

    cvm = cvm1_concordance_weighted(y, yhat)
    if not np.isfinite(cvm):
        return np.nan

    return 1 - cvm / g


# WRGR (robustness)
def wrgr_cramer(pred, pred_pert):
    g = gini_via_lorenz(pred)
    if not np.isfinite(g) or g == 0:
        return np.nan

    cvm = cvm1_concordance_weighted(pred, pred_pert)
    if not np.isfinite(cvm):
        return np.nan

    return 1 - cvm / g


# WRGE (explainability)
def wrge_cramer(pred_full, pred_reduced):
    g = gini_via_lorenz(pred_full)
    if not np.isfinite(g) or g == 0:
        return np.nan

    cvm = cvm1_concordance_weighted(pred_full, pred_reduced)
    if not np.isfinite(cvm):
        return np.nan

    return cvm / g


# Partial WRGA
def partial_wrga_cramer(y, yhat, lower=0.0, upper=1.0):
    """
    Function for computing partial WRGA.

    """
    y = np.asarray(y, float).reshape(-1)
    yhat = np.asarray(yhat, float).reshape(-1)

    df = pd.DataFrame({"y": y, "yhat": yhat})
    df = df.dropna().reset_index(drop=True)
    df["ryhat"] = df["yhat"].rank(method="min")

    support = df.groupby("ryhat")["y"].mean().reset_index(name="support")
    df = df.merge(support, on="ryhat", how="left")
    df["rord"] = df["support"]
    df = df.sort_values("yhat").reset_index(drop=True)
    ystar = df["rord"].values

    n = len(df)
    sorted_y = np.sort(df["y"].values)

    # Lorenz curve
    lorenz_y = np.insert(np.cumsum(sorted_y) / (n * sorted_y.mean()), 0, 0)
    x_grid = np.insert(np.linspace(1/n, 1, n), 0, 0)

    # Concordance curve
    concord_y = np.insert(np.cumsum(ystar) / (n * sorted_y.mean()), 0, 0)

    uniform_y = x_grid.copy()

    integrand_num = np.abs(concord_y - lorenz_y)
    integrand_den = np.abs(uniform_y - lorenz_y)
    total_denom = np.trapezoid(integrand_den, x_grid)

    if total_denom == 0:
        return np.nan

    # Slice
    mask = (x_grid >= lower) & (x_grid <= upper)
    x_slice = x_grid[mask]

    num_slice = np.trapezoid(integrand_num[mask], x_slice)
    den_slice = np.trapezoid(integrand_den[mask], x_slice)

    if den_slice == 0:
        return np.nan

    partial_wrga_contribution = num_slice / total_denom

    return partial_wrga_contribution



# Cramer test
def cvm_test_wrga(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 2 or len(y_pred) < 2:
        return dict(statistic=np.nan, pvalue=np.nan)

    test = cramervonmises_2samp(y_true, y_pred)
    return dict(statistic=test.statistic, pvalue=test.pvalue)