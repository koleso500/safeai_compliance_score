import numpy as np

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
def partial_wrga_cramer(y, yhat, n_segments):
    """
    Decompose WRGA into partial contributions across segments.

    Parameters:
    -----------
    y :
        True values
    yhat :
        Predicted values
    n_segments : int
        Number of segments to decompose into

    Returns:
    --------
    dict containing:
        - 'full_wrga': The complete WRGA score
        - 'partial_wrga': Partial WRGA contributions for each segment
        - 'cumulative_vector': Cumulative vector [WRGA, WRGA-WRGA_1, ..., 0]
        - 'segment_indices': List of index ranges for each segment

    """
    y = np.asarray(y, dtype=float).reshape(-1)
    yhat = np.asarray(yhat, dtype=float).reshape(-1)
    mask = ~np.isnan(y) & ~np.isnan(yhat)
    y = y[mask]
    yhat = yhat[mask]

    n = len(y)
    if n == 0:
        return {
            'full_wrga': np.nan,
            'partial_wrga': np.array([]),
            'cumulative_vector': np.array([]),
            'segment_indices': []
        }

    # Calculate full WRGA
    full_wrga = wrga_cramer(y, yhat)
    full_gini = gini_via_lorenz(y)

    if not np.isfinite(full_wrga) or not np.isfinite(full_gini) or full_gini == 0:
        return {
            'full_wrga': full_wrga,
            'partial_wrga': np.array([np.nan] * n_segments),
            'cumulative_vector': np.array([np.nan] * (n_segments + 1)),
            'segment_indices': []
        }

    # Sort by predictions (descending - from highest to lowest ranked)
    ord_yhat_desc = np.argsort(yhat)[::-1]
    y_sorted = y[ord_yhat_desc]
    yhat_sorted = yhat[ord_yhat_desc]

    # Divide into n_segments
    segment_size = n // n_segments
    remainder = n % n_segments

    partial_wrga = []
    segment_indices = []

    start_idx = 0
    for k in range(n_segments):
        # Remainder across first segments
        current_size = segment_size + (1 if k < remainder else 0)
        end_idx = start_idx + current_size

        segment_indices.append((start_idx, end_idx))

        # Extract segment
        y_segment = y_sorted[start_idx:end_idx]
        yhat_segment = yhat_sorted[start_idx:end_idx]

        # Calculate WRGA for this segment
        segment_wrga = wrga_cramer(y_segment, yhat_segment)

        # Weight by segment's contribution to total Gini
        segment_gini = gini_via_lorenz(y_segment)
        if np.isfinite(segment_gini) and segment_gini > 0:
            # Normalize by segment size relative to total
            weight = len(y_segment) / n
            weighted_contribution = segment_wrga * segment_gini * weight / full_gini
        else:
            weighted_contribution = 0.0

        partial_wrga.append(weighted_contribution)
        start_idx = end_idx

    partial_wrga = np.array(partial_wrga)

    # Normalize so sum equals full_wrga
    sum_partial = np.sum(partial_wrga)
    if sum_partial > 0:
        partial_wrga = partial_wrga * (full_wrga / sum_partial)

    # Build cumulative vector
    cumulative_vector = np.zeros(n_segments + 1)
    cumulative_vector[0] = full_wrga

    cumsum = 0.0
    for k in range(n_segments):
        cumsum += partial_wrga[k]
        cumulative_vector[k + 1] = full_wrga - cumsum

    return {
        'full_wrga': full_wrga,
        'partial_wrga': partial_wrga,
        'cumulative_vector': cumulative_vector,
        'segment_indices': segment_indices
    }


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