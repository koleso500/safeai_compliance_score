import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cramervonmises_2samp
from sklearn.metrics import auc


# Lorenz Curve
def lorenz_curve(y):
    """
    Compute the Lorenz curve for a given array.

    Parameters
    ----------
    y : array-like
        Input values

    Returns
    -------
    np.ndarray
        Normalized cumulative sum (Lorenz curve)
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    y = y[~np.isnan(y)]
    if len(y) == 0:
        return np.array([])
    y_sorted = np.sort(y)
    cum = np.cumsum(y_sorted)
    sum_y = cum[-1]
    if sum_y == 0:
        return np.full_like(cum, np.nan)
    return cum / sum_y


# Concordance Curve
def concordance_curve(y, yhat):
    """
    Compute the concordance curve between true and predicted values.

    Parameters
    ----------
    y : array-like
        True values
    yhat : array-like
        Predicted values

    Returns
    -------
    np.ndarray
        Concordance curve
    """
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
    """
    Calculate Gini coefficient.

    Parameters
    ----------
    y : array-like
        Input values

    Returns
    -------
    float
        Gini coefficient
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    l = lorenz_curve(y)
    n = len(l)
    if n == 0:
        return np.nan
    u = np.linspace(1 / n, 1, n)
    return 2 * np.mean(np.abs(u - l))


# Weighted Cramer (L1) distance between Lorenz and Concordance
def cvm1_concordance_weighted(y, yhat):
    """
    Weighted Cramer von Mises distance between Lorenz and Concordance curves.

    Parameters
    ----------
    y : array-like
        True values
    yhat : array-like
        Predicted values

    Returns
    -------
    float
        Weighted CvM distance
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    yhat = np.asarray(yhat, dtype=float).reshape(-1)
    mask = ~np.isnan(y) & ~np.isnan(yhat)
    y = y[mask]
    yhat = yhat[mask]

    n = len(y)
    if n == 0:
        return np.nan

    # Lorenz curve
    ord_y = np.argsort(y)
    l = np.cumsum(y[ord_y]) / np.sum(y)

    # Concordance curve
    ord_yhat = np.argsort(yhat)
    c = np.cumsum(y[ord_yhat]) / np.sum(y)

    # Weights
    weights = y[ord_y] / np.sum(y)

    return np.sum(np.abs(c - l) * weights)


# WRGA
def wrga_cramer(y, yhat):
    """
    Weighted RGA (WRGA) using Cramer distance.
    WRGA = 1 - CvM(y, yhat) / G(y)

    Parameters
    ----------
    y : array-like
        True values
    yhat : array-like
        Predicted values

    Returns
    -------
    float
        WRGA score
    """
    g = gini_via_lorenz(y)
    if not np.isfinite(g) or g == 0:
        return np.nan

    cvm = cvm1_concordance_weighted(y, yhat)
    if not np.isfinite(cvm):
        return np.nan

    return 1 - cvm / g


def wrgr_cramer(pred, pred_pert):
    """
    Weighted RGR (WRGR) which compares original predictions with perturbed predictions.

    Parameters
    ----------
    pred : array-like
        Original predictions
    pred_pert : array-like
        Perturbed predictions

    Returns
    -------
    float
        WRGR score
    """
    g = gini_via_lorenz(pred)
    if not np.isfinite(g) or g == 0:
        return np.nan
    cvm = cvm1_concordance_weighted(pred, pred_pert)
    if not np.isfinite(cvm):
        return np.nan
    return 1 - cvm / g


def wrge_cramer(pred, pred_reduced):
    """
    Weighted RGE (WRGE) which compares original predictions with perturbed predictions.

    Parameters
    ----------
    pred : array-like
        Predictions from full model
    pred_reduced : array-like
        Predictions from reduced model

    Returns
    -------
    float
        WRGE score
    """
    g = gini_via_lorenz(pred)
    if not np.isfinite(g) or g == 0:
        return np.nan
    cvm = cvm1_concordance_weighted(pred, pred_reduced)
    if not np.isfinite(cvm):
        return np.nan
    return cvm / g


# Partial WRGA
def partial_wrga_cramer(y, yhat, n_segments):
    """
    Decompose WRGA into partial contributions across segments.

    Parameters
    ----------
    y : array-like
        True values
    yhat : array-like
        Predicted values
    n_segments : int
        Number of segments to decompose into

    Returns
    -------
    dict
        Dictionary containing:
        - 'full_wrga': WRGA score
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

    # Sort by predictions (descending)
    ord_yhat_desc = np.argsort(yhat)[::-1]
    y_sorted = y[ord_yhat_desc]
    yhat_sorted = yhat[ord_yhat_desc]

    # Divide into segments
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

    # Normalize
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


# Multiclass WRGA
def wrga_cramer_multiclass(y_labels, prob_matrix, class_order=None, verbose=False):
    """
    Calculate WRGA for multiclass classification using one-vs-rest approach.

    Parameters
    ----------
    y_labels : array-like
        True class labels
    prob_matrix : array-like, shape (n_samples, n_classes)
        Predicted probabilities for each class.
        Columns must correspond to `class_order` if provided,
        or to sorted unique classes in y_labels if not.
    class_order : array-like, optional
        Order of classes corresponding to prob_matrix columns (.classes_).
        If None, assumes prob_matrix columns match sorted unique(y_labels).
    verbose : bool, optional
        Print detailed information

    Returns
    -------
    tuple
        (wrga_weighted, wrga_per_class, class_weights, classes_used)
        - wrga_weighted: Overall weighted WRGA score
        - wrga_per_class: WRGA score for each class
        - class_weights: Weight of each class
        - classes_used: The class order used for computation
    """
    y_labels = np.asarray(y_labels)
    prob_matrix = np.asarray(prob_matrix)

    # Determine class order
    if class_order is None:
        if verbose:
            print('WARNING: class_order is not provided. Assuming prob_matrix columns match sorted unique classes.')
        class_order = np.unique(y_labels)
    else:
        class_order = np.asarray(class_order)

    n_classes = len(class_order)

    # Validate dimensions
    if prob_matrix.shape[1] != n_classes:
        raise ValueError(
            f'prob_matrix has {prob_matrix.shape[1]} columns but class_order has {n_classes} classes.'
        )

    wrgas = []
    weights = []

    for k, c in enumerate(class_order):
        # One-vs-rest encoding
        y_bin = np.equal(y_labels, c).astype(np.float32)
        yhat_c = prob_matrix[:, k]

        if np.sum(y_bin) == 0:
            if verbose:
                print(f'Warning: Class {c} has zero samples. Skipping.')
            wrgas.append(0.0)
            weights.append(0.0)
            continue

        wrga_k = wrga_cramer(y_bin, yhat_c)
        wrgas.append(wrga_k)
        weights.append(np.mean(y_bin))

    wrgas = np.array(wrgas)
    weights = np.array(weights)

    # Weighted average
    wrga_weighted = np.nansum(wrgas * weights) / np.nansum(weights)

    return wrga_weighted, wrgas, weights, class_order


def partial_wrga_cramer_multiclass(y_labels, prob_matrix, n_segments, class_order=None, verbose=False):
    """
    Calculate partial WRGA curves for multiclass classification.

    Parameters
    ----------
    y_labels : array-like
        True class labels
    prob_matrix : array-like, shape (n_samples, n_classes)
        Predicted probabilities for each class.
    n_segments : int
        Number of segments for partial decomposition
    class_order : array-like, optional
        Order of classes corresponding to prob_matrix columns.
    verbose : bool, optional
        Print detailed information

    Returns
    -------
    dict
        Dictionary containing:
        - 'cumulative_vector': Weighted average cumulative vector
        - 'per_class_vectors': Cumulative vectors for each class
        - 'class_weights': Weight of each class
        - 'classes': Class order used
    """
    y_labels = np.asarray(y_labels)
    prob_matrix = np.asarray(prob_matrix)

    # Determine class order
    if class_order is None:
        if verbose:
            print('WARNING: class_order is not provided. Assuming prob_matrix columns match sorted unique classes.')
        class_order = np.unique(y_labels)
    else:
        class_order = np.asarray(class_order)

    cum_vectors = []
    class_weights = []

    for k, c in enumerate(class_order):
        y_bin = np.equal(y_labels, c).astype(np.float32)
        yhat_c = prob_matrix[:, k]

        res = partial_wrga_cramer(y_bin, yhat_c, n_segments)
        cum_vectors.append(res['cumulative_vector'])
        class_weights.append(np.mean(y_bin))

    cum_vectors = np.vstack(cum_vectors)
    class_weights = np.array(class_weights)

    # Weighted average across classes
    weighted_curve = np.average(cum_vectors, weights=class_weights, axis=0)

    return {
        'cumulative_vector': weighted_curve,
        'per_class_vectors': cum_vectors,
        'class_weights': class_weights,
        'classes': class_order
    }


# Evaluation Function
def evaluate_wrga_multiclass(y_labels, prob_matrix, class_order=None, n_segments=10,
                             model_name='Model', plot=True,
                             fig_size=(12, 5), verbose=True):
    """
    WRGA evaluation for multiclass classification.

    Parameters
    ----------
    y_labels : array-like
        True class labels
    prob_matrix : array-like, shape (n_samples, n_classes)
        Predicted probabilities for each class
    class_order : array-like, optional
        Order of classes corresponding to prob_matrix columns.
        For sklearn models, pass `model.classes_`.
        For PyTorch models, pass the class order used in output layer, like np.array([0, 1, 2, ...]).
    n_segments : int, optional
        Number of segments for partial WRGA decomposition
    model_name : str, optional
        Name of the model for display
    plot : bool, optional
        Whether to generate visualization
    fig_size : tuple, optional
        Figure size for plots
    verbose : bool, optional
        Print detailed results

    Returns
    -------
    dict
        Comprehensive results dictionary containing:
        - 'wrga_full': Overall WRGA score
        - 'wrga_per_class': WRGA for each class
        - 'class_weights': Weight of each class
        - 'aurga': Area under RGA curve
        - 'cumulative_vector': Cumulative WRGA vector
        - 'per_class_vectors': Per-class cumulative vectors
        - 'classes': Class order used
    """
    # Calculate full WRGA
    wrga_full, wrga_per_class, class_weights, classes_used = wrga_cramer_multiclass(
        y_labels, prob_matrix, class_order=class_order, verbose=verbose
    )

    # Calculate partial WRGA
    partial_results = partial_wrga_cramer_multiclass(
        y_labels, prob_matrix, n_segments, class_order=class_order, verbose=verbose
    )

    cumulative_vector = partial_results['cumulative_vector']
    x_axis = np.linspace(0, 1, len(cumulative_vector))

    # Calculate AURGA
    aurga = auc(x_axis, cumulative_vector)

    # Print results
    if verbose:
        print(f'WRGA EVALUATION: {model_name}')
        print(f'Full RGA: {wrga_full:.4f}')
        print(f'AURGA: {aurga:.4f}')
        print(f'\nClass order: {classes_used}')
        print('\nPer-Class RGA:')
        for i, (cls, wrga_val, weight) in enumerate(
                zip(classes_used, wrga_per_class, class_weights)
        ):
            print(f'Class {cls}: RGA={wrga_val:.4f}, Weight={weight:.4f}')

    # Visualization
    if plot:
        plt.figure(figsize=fig_size)
        plt.plot(x_axis, cumulative_vector, marker='o', linewidth=2.5, markersize=6, color='steelblue',
            label=f'{model_name} (AURGA={aurga:.3f})'
        )

        plt.fill_between(x_axis,0, cumulative_vector, alpha=0.2, color='steelblue')
        plt.xlabel('Fraction of Data Removed', fontsize=11, fontweight='bold')
        plt.ylabel('RGA Score', fontsize=11, fontweight='bold')
        plt.title('RGA Robustness Curve', fontsize=12, fontweight='bold')
        plt.grid(alpha=0.3, linestyle='--')
        plt.xlim([0, 1])
        max_val = np.nanmax(cumulative_vector)
        plt.ylim([0, max_val * 1.1 if np.isfinite(max_val) else 1])
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()

    return {
        'wrga_full': wrga_full,
        'wrga_per_class': wrga_per_class,
        'class_weights': class_weights,
        'aurga': aurga,
        'cumulative_vector': cumulative_vector,
        'per_class_vectors': partial_results['per_class_vectors'],
        'classes': classes_used
    }


def compare_models_wrga(models_dict, y_labels, n_segments=10,
                        fig_size=(14, 6), verbose=True):
    """
    Compare multiple models using WRGA metrics.

    Parameters
    ----------
    models_dict : dict
        Dictionary mapping model names to tuples of (prob_matrix, class_order).
        Example: {
            'Random Forest': (rf.predict_proba(x_test), rf.classes_),
            'Neural Network': (nn_probs, np.array([0, 1, 2]))
        }
    y_labels : array-like
        True class labels
    n_segments : int, optional
        Number of segments for partial WRGA
    fig_size : tuple, optional
        Figure size for comparison plot
    verbose : bool, optional
        Print detailed comparison

    Returns
    -------
    dict
        Comparison results for all models
    """
    results = {}

    for model_name, (prob_matrix, class_order) in models_dict.items():
        if verbose:
            print(f'\nEvaluating {model_name}...')

        result = evaluate_wrga_multiclass(
            y_labels, prob_matrix, class_order=class_order,
            n_segments=n_segments, model_name=model_name,
            plot=False, verbose=verbose
        )
        results[model_name] = result

    # Comparison plot
    plt.figure(figsize=fig_size)

    cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, len(results)))

    for (model_name, result), color in zip(results.items(), colors):
        x_axis = np.linspace(0, 1, len(result['cumulative_vector']))
        plt.plot(x_axis, result['cumulative_vector'], "-o", linewidth=2.5, markersize=5,
            color=color, label=f"{model_name} (AURGA={result['aurga']:.3f})"
        )
    plt.xlabel('Fraction of Data Removed', fontsize=11, fontweight='bold')
    plt.ylabel('RGA Score', fontsize=11, fontweight='bold')
    plt.title('RGA Curve Comparison', fontsize=12, fontweight='bold')
    plt.grid(alpha=0.3, linestyle="--")
    plt.xlim([0, 1])
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.show()

    if verbose:
        model_names = list(results.keys())
        aurga_scores = [results[n]['aurga'] for n in model_names]
        wrga_scores = [results[n]['rga_full'] for n in model_names]

        print('RGA Comparison Summary')
        for n, w, a in zip(model_names, wrga_scores, aurga_scores):
            print(f'{n}: RGA={w:.4f}, AURGA={a:.4f}')

    return results


# Statistical Testing
def cvm_test_wrga(y_true, y_pred):
    """
    Cramer von Mises two-sample test for WRGA.

    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values

    Returns
    -------
    dict
        Test statistic and p-value
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 2 or len(y_pred) < 2:
        return dict(statistic=np.nan, pvalue=np.nan)

    test = cramervonmises_2samp(y_true, y_pred)
    return dict(statistic=test.statistic, pvalue=test.pvalue)