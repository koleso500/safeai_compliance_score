import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
import torch
from typing import Union, Literal
from sklearn.metrics import auc
from safeai_files.utils import check_nan, convert_to_dataframe, find_yhat, validate_variables
from xgboost import XGBClassifier, XGBRegressor

from safeai_files.core import rga
from safeai_files.cramer import wrga_cramer, wrgr_cramer


def perturb(data: pd.DataFrame, 
            variable: str, 
            perturbation_percentage= 0.05):
    """
    Function to perturb a single variable based on the replacement of the two percentiles 
    selected using the perturbation_percentage of the object.

    Parameters
    ----------
    data : pd.DataFrame
            A dataframe including data.
    variable: str 
            Name of variable.
    perturbation_percentage: float
            A percentage value for perturbation. 

    Returns
    -------
    pd.DataFrame
            The perturbed data.
    """ 
    if perturbation_percentage > 0.5 or perturbation_percentage < 0:
        raise ValueError('The perturbation percentage should be between 0 and 0.5.')
        
    data = data.reset_index(drop=True)
    perturbed_variable = data.loc[:,variable]
    vals = [[i, values] for i, values in enumerate(perturbed_variable)]
    indices = [x[0] for x in sorted(vals, key= lambda item: item[1])]
    sorted_variable = [x[1] for x in sorted(vals, key= lambda item: item[1])]

    percentile_5_index = int(np.ceil(perturbation_percentage * len(sorted_variable)))
    percentile_95_index = int(np.ceil((1-perturbation_percentage) * len(sorted_variable)))
    values_before_5th_percentile = sorted_variable[:percentile_5_index]
    values_after_95th_percentile = sorted_variable[percentile_95_index:]
    n = min([len(values_before_5th_percentile), len(values_after_95th_percentile)])
    lower_tail_indices = indices[0:n]
    upper_tail_indices = (indices[-n:])
    upper_tail_indices = upper_tail_indices[::-1]
    new_variable = perturbed_variable.copy()

    for j in range(n):
        new_variable[lower_tail_indices[j]] = perturbed_variable[upper_tail_indices[j]]
        new_variable[upper_tail_indices[j]] = perturbed_variable[lower_tail_indices[j]]
    data.loc[:,variable] = new_variable
    return data


def compute_rgr_values(xtest: pd.DataFrame, 
                                yhat: list, 
                                model: Union[XGBClassifier, XGBRegressor, BaseEstimator,
                                torch.nn.Module],
                                variables: list, 
                                perturbation_percentage= 0.05):
    """
    Compute RANK GRADUATION Robustness (RGR) MEASURE for single variable contribution.

    Parameters
    ----------
    xtest : pd.DataFrame
            A dataframe including test data.
    yhat : list
            A list of predicted values.
    model : Union[XGBClassifier, XGBRegressor, BaseEstimator, torch.nn.Module]
            A trained model, which could be a classifier or regressor. 
    variables: list 
            A list of variables.
    perturbation_percentage: float
            A percentage value for perturbation .

    Returns
    -------
    pd.DataFrame
            The RGR value.
    """
    # Convert inputs to DataFrames and concatenate them
    xtest, yhat = convert_to_dataframe(xtest, yhat)
    # check for missing values
    check_nan(xtest, yhat)
    # variables should be a list
    validate_variables(variables, xtest)
    # find RGRs
    rgr_list = []
    for i in variables:
        xtest_pert = perturb(xtest, i, perturbation_percentage)
        yhat_pert = find_yhat(model, xtest_pert)
        rgr_list.append(rga(yhat, yhat_pert))
    rgr_df = pd.DataFrame(rgr_list, index= list(variables), columns=['RGR']).sort_values(by='RGR', ascending=False)
    return rgr_df


def rgr_single(xtest: pd.DataFrame,
                yhat: list,
                model: Union[XGBClassifier, XGBRegressor, BaseEstimator, torch.nn.Module],
                variable: str,
                perturbation_percentage=0.05):
    """
    Compute RANK GRADUATION Robustness (RGR) MEASURE for a single variable.

    Parameters
    ----------
    xtest : pd.DataFrame
            A dataframe including test data.
    yhat : list
            A list of predicted values.
    model : Union[XGBClassifier, XGBRegressor, BaseEstimator, torch.nn.Module]
            A trained model, which could be a classifier or regressor.
    variable : str
            The variable (column) in xtest to be perturbed.
    perturbation_percentage: float
            A percentage value for perturbation.

    Returns
    -------
    float
            The RGR value for the specified variable.
    """
    # Convert inputs to DataFrames and concatenate them
    xtest, yhat = convert_to_dataframe(xtest, yhat)

    # Check for missing values
    check_nan(xtest, yhat)

    # Variables should be a list
    validate_variables(variable, xtest)

    # Perturb only the selected variable
    xtest_pert = xtest.copy()
    xtest_pert[variable] = perturb(xtest_pert, variable, perturbation_percentage)[variable]

    # Get perturbed predictions
    yhat_pert = find_yhat(model, xtest_pert)

    # Compute and return RGR value for the selected variable
    return rga(yhat, yhat_pert)



def rgr_all(xtest: pd.DataFrame,
            yhat: list,
            model: Union[XGBClassifier,
            XGBRegressor, BaseEstimator, torch.nn.Module],
            perturbation_percentage = 0.05,
            perturbation_strategy: Literal['percentile_swap', 'gaussian_noise'] = 'percentile_swap',
            seed: int = None,
            scaler = None,
            xtest_original: pd.DataFrame = None,
            metric: str = 'original'):
    """
    Compute RANK GRADUATION Robustness (RGR) MEASURE for all variables simultaneously.

    Parameters
    ----------
    xtest : pd.DataFrame
        A dataframe including test data.
    yhat : list
        A list of predicted values.
    model : Union[XGBClassifier, XGBRegressor,
                  BaseEstimator, torch.nn.Module]
        A trained model, which could be a classifier or regressor.
    perturbation_percentage : float
        A percentage value for perturbation intensity.
        - For 'percentile_swap': percentage of tail values to swap
        - For 'gaussian_noise': noise standard deviation as fraction of entire data std
    perturbation_strategy : str
        - 'percentile_swap': Swap tail percentiles
        - 'gaussian_noise': Add Gaussian noise to data
    seed : int
        Random seed for reproducibility (only with 'gaussian_noise')
    scaler :
        Use if features are scaled. Noise is added in original scale then re-scaled.
    xtest_original : pd.DataFrame
        Original unscaled test data. Required when scaler.
    metric: str
        'original': uses RGE
        'cramer': uses WRGE

    Returns
    -------
    float
        The overall RGR value.

    """
    if metric not in ['original', 'cramer']:
        raise ValueError("Metric must be 'original' or 'cramer'")

    # Convert inputs to DataFrames and concatenate them
    xtest, yhat = convert_to_dataframe(xtest, yhat)

    # Check for missing values
    check_nan(xtest, yhat)

    # Get all variables in xtest
    variables = xtest.columns.tolist()

    # Perturb based on strategy
    if perturbation_strategy == 'percentile_swap':
        xtest_pert = xtest.copy()
        for var in variables:
            xtest_pert[var] = perturb(xtest_pert, var, perturbation_percentage)[var]

    elif perturbation_strategy == 'gaussian_noise':
        if seed is not None:
            np.random.seed(seed)

        if scaler is not None and xtest_original is not None:
            # Add noise in original scale, then re-scale
            base_std = np.std(xtest_original.values)
            noise = np.random.normal(0.0, perturbation_percentage * base_std,
                                     size=xtest_original.shape).astype(np.float32)

            # Add noise to original data
            xtest_perturbed_original = xtest_original.values + noise

            xtest_perturbed_df = pd.DataFrame(xtest_perturbed_original,
                                              columns=xtest_original.columns,
                                              index=xtest_original.index)

            # Re-scale the perturbed data
            xtest_pert_array = scaler.transform(xtest_perturbed_df)

        else:
            # Calculate base std
            base_std = np.std(xtest.values)

            # Generate Gaussian noise
            noise = np.random.normal(0.0, perturbation_percentage * base_std,
                                    size=xtest.shape).astype(np.float32)

            # Add noise to features
            xtest_pert_array = xtest.values + noise

        # Convert back to DataFrame
        xtest_pert = pd.DataFrame(xtest_pert_array,
                                  columns=xtest.columns,
                                  index=xtest.index)

    else:
        raise ValueError(f'Unknown perturbation strategy: {perturbation_strategy}.'
                         f"Choose from: 'percentile_swap', 'gaussian_noise'")

    # Get perturbed predictions
    yhat_pert = find_yhat(model, xtest_pert)

    # Compute and return RGR value
    if metric == "original":
        return rga(yhat, yhat_pert)
    else:  # 'cramer'
        return wrga_cramer(yhat, yhat_pert)


def align_proba_to_class_order(prob, model_class_order, target_class_order):
    """
    Align probability matrix columns to match a target class order.

    Parameters
    ----------
    prob : array-like, shape (n_samples, n_classes)
        Probability matrix with columns in model_class_order
    model_class_order : array-like
        Current order of classes (e.g., model.classes_ for sklearn)
    target_class_order : array-like
        Desired order of classes

    Returns
    -------
    np.ndarray
        Probability matrix with columns reordered to match target_class_order

    Examples
    --------
    prob_aligned = align_proba_to_class_order(model.predict_proba(x), model.classes_, [0, 1, 2])
    """
    prob = np.asarray(prob)
    model_class_order = list(model_class_order)
    target_class_order = list(target_class_order)

    # Find the index mapping
    idx = [model_class_order.index(c) for c in target_class_order]

    return prob[:, idx]



# Multiclass WRGR
def wrgr_cramer_multiclass(prob_original, prob_perturbed, class_weights=None, verbose=False):
    """
    Calculate WRGR for multiclass classification.

    Parameters
    ----------
    prob_original : array-like, shape (n_samples, n_classes)
        Original predicted probabilities
    prob_perturbed : array-like, shape (n_samples, n_classes)
        Perturbed predicted probabilities
    class_weights : array-like, optional
        Custom weights for each class. If None, uses uniform weighting.
    verbose : bool, optional
        Print detailed information

    Returns
    -------
    tuple
        (wrgr_weighted, wrgr_per_class, weights_used)
        - wrgr_weighted: Overall weighted WRGR score
        - wrgr_per_class: WRGR score for each class
        - weights_used: Weights used for each class
    """
    prob_original = np.asarray(prob_original)
    prob_perturbed = np.asarray(prob_perturbed)

    n_samples, n_classes = prob_original.shape

    if prob_perturbed.shape != prob_original.shape:
        raise ValueError(
            f'Shape mismatch: prob_original {prob_original.shape} and prob_perturbed {prob_perturbed.shape}'
        )

    # Set up class weights
    if class_weights is None:
        class_weights = np.ones(n_classes) / n_classes
    else:
        class_weights = np.asarray(class_weights)
        if len(class_weights) != n_classes:
            raise ValueError(
                f'Class_weights length {len(class_weights)} does not match n_classes {n_classes}'
            )

    wrgrs = []

    for k in range(n_classes):
        pred_orig = prob_original[:, k]
        pred_pert = prob_perturbed[:, k]

        wrgr_k = wrgr_cramer(pred_orig, pred_pert)
        wrgrs.append(wrgr_k)

        if verbose:
            print(f'Class {k}: WRGR = {wrgr_k:.4f}')

    wrgrs = np.array(wrgrs)

    # Weighted average
    wrgr_weighted = np.nansum(wrgrs * class_weights) / np.nansum(class_weights)

    return wrgr_weighted, wrgrs, class_weights


def evaluate_wrgr_multiclass_noise(model, x_data, prob_original, noise_levels,
                                   model_class_order, class_order,
                                   class_weights=None, model_type='sklearn',
                                   device=None, wrga_full=None, model_name="Model",
                                   plot=True, fig_size=(10, 6), verbose=True,
                                   random_seed=None):
    """
    Evaluate WRGR robustness for multiclass classification with noise perturbation.

    Parameters
    ----------
    model : sklearn model or PyTorch model
        Trained model to evaluate
    x_data : array-like or torch.Tensor
        Input features
    prob_original : array-like, shape (n_samples, n_classes)
        Original predicted probabilities (columns in model_class_order)
    noise_levels : array-like
        Standard deviations of Gaussian noise to test
    model_class_order : array-like
        Order of classes in model's output (e.g., model.classes_ for sklearn)
    class_order : array-like
        Target class order for alignment (shared across all models in comparison)
    class_weights : array-like, optional
        Custom weights for each class. If None, uses uniform weighting.
    model_type : {'sklearn', 'pytorch'}, optional
        Type of model
    device : torch.device, optional
        Device for PyTorch models
    wrga_full : float, optional
        Full WRGA score for rescaling. If None, no rescaling is applied.
    model_name : str, optional
        Name of model for display
    plot : bool, optional
        Whether to generate visualization
    fig_size : tuple, optional
        Figure size for plot
    verbose : bool, optional
        Print detailed results
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict
        Dictionary containing:
        - 'wrgr_scores': WRGR scores at each noise level
        - 'wrgr_rescaled': Rescaled WRGR scores (if wrga_full provided)
        - 'aurgr': Area under RGR curve
        - 'noise_levels': Noise levels tested
        - 'per_class_wrgr': Per-class WRGR at each noise level
        - 'class_order': Class order used
    """
    prob_original = np.asarray(prob_original)
    noise_levels = np.asarray(noise_levels)
    model_class_order = np.asarray(model_class_order)
    class_order = np.asarray(class_order)

    # Set random seed for reproducibility
    rng = np.random.default_rng(random_seed)

    # Align prob_original to target class_order at start
    prob_original_aligned = align_proba_to_class_order(
        prob_original, model_class_order, class_order
    )

    n_samples, n_classes = prob_original_aligned.shape

    # Set up class weights
    if class_weights is None:
        class_weights = np.ones(n_classes) / n_classes

    # Convert PyTorch tensor once if needed
    if model_type == 'pytorch':
        import torch
        if not isinstance(x_data, torch.Tensor):
            x_data = torch.tensor(x_data, dtype=torch.float32)

    wrgr_scores = []
    per_class_wrgr_list = []

    if verbose:
        print(f'RGR Evaluation: {model_name}')
        print(f'Testing {len(noise_levels)} noise levels')

    for i, sigma in enumerate(noise_levels):
        if model_type == 'sklearn':
            # Add Gaussian noise to features
            noise = rng.normal(0, sigma, size=x_data.shape)
            x_noisy = x_data + noise

            # Get predictions and align to target class order
            prob_perturbed_raw = model.predict_proba(x_noisy)
            prob_perturbed = align_proba_to_class_order(
                prob_perturbed_raw, model_class_order, class_order
            )

        elif model_type == 'pytorch':
            import torch
            # Add Gaussian noise to tensor
            noise = torch.randn_like(x_data) * sigma
            x_noisy = x_data + noise

            # Get predictions and align to target class order
            with torch.no_grad():
                logits = model(x_noisy.to(device))
                prob_perturbed_raw = torch.softmax(logits, dim=1).cpu().numpy()

            prob_perturbed = align_proba_to_class_order(
                prob_perturbed_raw, model_class_order, class_order
            )
        else:
            raise ValueError(f"model_type must be 'sklearn' or 'pytorch', got '{model_type}'")

        # Calculate WRGR
        wrgr_val, wrgr_per_class, _ = wrgr_cramer_multiclass(
            prob_original_aligned,
            prob_perturbed,
            class_weights=class_weights,
            verbose=False
        )

        wrgr_scores.append(0.0 if np.isnan(wrgr_val) else wrgr_val)
        per_class_wrgr_list.append(wrgr_per_class)

        if verbose and i % max(1, len(noise_levels) // 5) == 0:
            print(f"σ = {sigma:.3f}: RGR = {wrgr_scores[-1]:.4f}")

    wrgr_scores = np.array(wrgr_scores)
    per_class_wrgr_list = np.array(per_class_wrgr_list)

    # Rescale by full WRGA if provided
    if wrga_full is not None:
        wrgr_rescaled = wrgr_scores * wrga_full
    else:
        wrgr_rescaled = wrgr_scores

    # Normalize noise for AUC calculation
    max_noise = np.max(noise_levels)
    noise_norm = noise_levels / max_noise if max_noise > 0 else noise_levels

    # Calculate AURGR
    aurgr = auc(noise_norm, wrgr_rescaled)

    if verbose:
        print(f'AURGR: {aurgr:.4f}')

    # Visualization
    if plot:
        plt.figure(figsize=fig_size)
        plt.plot(noise_levels * 100, wrgr_rescaled, '-o', linewidth=2.5,
                 markersize=6, color='steelblue',
                 label=f'{model_name} (AURGR={aurgr:.3f})')
        plt.fill_between(noise_levels * 100, 0, wrgr_rescaled,
                         alpha=0.2, color='steelblue')
        plt.xlabel('Noise Standard Deviation', fontsize=11, fontweight='bold')
        plt.ylabel('RGR Score', fontsize=11, fontweight='bold')
        plt.title(f'RGR Curve: {model_name}', fontsize=12, fontweight='bold')
        plt.grid(alpha=0.3, linestyle='--')
        plt.xlim([0, noise_levels[-1] * 100])
        plt.ylim([0, max(wrgr_rescaled) * 1.1 if max(wrgr_rescaled) > 0 else 1])
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()

    return {
        'wrgr_scores': wrgr_scores,
        'wrgr_rescaled': wrgr_rescaled,
        'aurgr': aurgr,
        'noise_levels': noise_levels,
        'per_class_wrgr': per_class_wrgr_list,
        'class_order': class_order
    }


def compare_models_wrgr(models_dict, noise_levels, class_order,
                        wrga_dict=None, class_weights=None,
                        fig_size=(12, 6), verbose=True, random_seed=None):
    """
    Compare robustness of multiple models using WRGR metrics.

    Parameters
    ----------
    models_dict : dict
        Dictionary mapping model names to tuples of:
        (model, X_data, prob_original, model_class_order, model_type, device)

        Example: {
            'RF': (rf_model, X_test, prob_rf, rf.classes_, 'sklearn', None),
            'VQC': (vqc_model, X_tensor, prob_vqc, np.array([0,1,2]), 'pytorch', device)
        }
    noise_levels : array-like
        Standard deviations of noise to test
    class_order : array-like
        Target class order for alignment (shared across all models)
    wrga_dict : dict, optional
        Dictionary mapping model names to WRGA scores for rescaling
    class_weights : array-like, optional
        Class weights for all models
    fig_size : tuple, optional
        Figure size for comparison plot
    verbose : bool, optional
        Print detailed results
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict
        WRGR evaluation results for all models
    """
    results = {}

    for model_name, model_config in models_dict.items():
        model, x_data, prob_original, model_class_order, model_type, device = model_config
        wrga_full = wrga_dict.get(model_name) if wrga_dict else None

        if verbose:
            print(f'\nEvaluating {model_name}...')

        result = evaluate_wrgr_multiclass_noise(
            model=model,
            x_data=x_data,
            prob_original=prob_original,
            noise_levels=noise_levels,
            model_class_order=model_class_order,
            class_order=class_order,
            class_weights=class_weights,
            model_type=model_type,
            device=device,
            wrga_full=wrga_full,
            model_name=model_name,
            plot=False,
            verbose=verbose,
            random_seed=random_seed
        )
        results[model_name] = result

    model_names = list(results.keys())
    aurgr_scores = np.array([results[name]['aurgr'] for name in model_names], dtype=float)

    plt.figure(figsize=fig_size)
    cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, len(results)))

    for (model_name, result), color in zip(results.items(), colors):
        plt.plot(
            result['noise_levels'] * 100,
            result['wrgr_rescaled'],
            '-o',
            linewidth=2.5,
            markersize=5,
            color=color,
            label=f"{model_name} (AURGR={result['aurgr']:.3f})"
        )

    plt.xlabel('Noise Standard Deviation', fontsize=11, fontweight='bold')
    plt.ylabel('RGR Score', fontsize=11, fontweight='bold')
    plt.title('RGR Curves Comparison', fontsize=12, fontweight='bold')
    plt.grid(alpha=0.3, linestyle='--')
    plt.xlim([0, float(np.max(noise_levels)) * 100])
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.show()

    if verbose:
        print('Robustness Comparison Summary')
        for name, score in zip(model_names, aurgr_scores):
            print(f'{name}: AURGR = {score:.4f}')

        if len(model_names) >= 2:
            best_idx = int(np.nanargmax(aurgr_scores))
            worst_idx = int(np.nanargmin(aurgr_scores))

            best = aurgr_scores[best_idx]
            worst = aurgr_scores[worst_idx]

            print(f'Best: {model_names[best_idx]} (AURGR={best:.4f})')
            print(f'Worst: {model_names[worst_idx]} (AURGR={worst:.4f})')

    return results


