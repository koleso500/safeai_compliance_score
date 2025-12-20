import pandas as pd
from sklearn.base import BaseEstimator
import torch
from typing import Union
from safeai_files.utils import check_nan, convert_to_dataframe, find_yhat, manipulate_testdata, validate_variables
from xgboost import XGBClassifier, XGBRegressor
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import auc
import random
import gc

from safeai_files.core import rga
from safeai_files.check_robustness import align_proba_to_class_order
from safeai_files.cramer import wrga_cramer, wrgr_cramer

def compute_rge_values(xtrain: pd.DataFrame, 
                xtest: pd.DataFrame,
                yhat: list,
                model: Union[XGBClassifier, XGBRegressor, BaseEstimator,
                torch.nn.Module],
                variables: list, 
                group: bool = False,
                metric: str = 'original'):
    """
    Helper function to compute the RGE values for given variables or groups of variables.

    Parameters
    ----------
    xtrain : pd.DataFrame
            A dataframe including train data.
    xtest : pd.DataFrame
            A dataframe including test data.
    yhat : list
            A list of predicted values.
    model : Union[XGBClassifier, XGBRegressor, BaseEstimator, torch.nn.Module]
            A trained model, which could be a classifier or regressor. 
    variables : list
            A list of variables.
    group : bool
            If True, calculate RGE for the group of variables as a whole; otherwise, calculate for each variable.
    metric: str
            'original': uses RGE
            'cramer': uses WRGE

    Returns
    -------
    pd.DataFrame
            The RGE values for each variable or for the group.
    """
    # Convert inputs to DataFrames and concatenate them
    xtrain, xtest, yhat = convert_to_dataframe(xtrain, xtest, yhat)
    # check for missing values
    check_nan(xtrain, xtest, yhat)
    # variables should be a list
    validate_variables(variables, xtrain)

    if metric not in ['original', 'cramer']:
        raise ValueError("Metric must be 'original' or 'cramer'")

    # find RGEs
    if group:
        # Apply manipulate_testdata iteratively for each variable in the group
        for variable in variables:
            xtest = manipulate_testdata(xtrain, xtest, variable)
        
        # Calculate yhat after manipulating all variables in the group
        yhat_rm = find_yhat(model, xtest)

        # Calculate a single RGE for the entire group except these variables
        if metric == "original":
            rge = rga(yhat, yhat_rm)
        else: # 'cramer'
            rge = wrga_cramer(yhat, yhat_rm)

        return pd.DataFrame([rge], index=[str(variables)], columns=['RGE'])

    else:
        # Calculate RGE for each variable individually
        rge_list = []
        for variable in variables:
            xtest_rm = manipulate_testdata(xtrain, xtest, variable)
            yhat_rm = find_yhat(model, xtest_rm)

            if metric == "original":
                rge_val = 1 - rga(yhat, yhat_rm)
            else:  # "cramer"
                rge_val = 1 - wrga_cramer(yhat, yhat_rm)

            rge_list.append(rge_val)
        
        return pd.DataFrame(rge_list, index=variables, columns=['RGE']).sort_values(by='RGE', ascending=False)


def wrge_cramer_multiclass(prob_full, prob_reduced, class_weights=None, verbose=False):
    """
    Calculate WRGE for multiclass classification.
    Measures impact of feature removal/occlusion on predictions.
    Use align_proba_to_class_order() before calling this function.

    Parameters
    ----------
    prob_full : array-like, shape (n_samples, n_classes)
        Predictions from original model
    prob_reduced : array-like, shape (n_samples, n_classes)
        Predictions from occluded model
    class_weights : array-like, optional
        Custom weights for each class. If None, uses uniform weighting.
    verbose : bool, optional
        Print detailed information

    Returns
    -------
    tuple
        (wrge_weighted, wrge_per_class, weights_used)
        - wrge_weighted: Overall weighted WRGE score
        - wrge_per_class: WRGE score for each class
        - weights_used: Weights used for each class
    """
    prob_full = np.asarray(prob_full)
    prob_reduced = np.asarray(prob_reduced)

    n_samples, n_classes = prob_full.shape

    if prob_reduced.shape != prob_full.shape:
        raise ValueError(
            f'Shape mismatch: prob_full {prob_full.shape} and prob_reduced {prob_reduced.shape}'
        )

    # Set up class weights
    if class_weights is None:
        class_weights = np.ones(n_classes) / n_classes
    else:
        class_weights = np.asarray(class_weights)
        if len(class_weights) != n_classes:
            raise ValueError(
                f'class_weights length {len(class_weights)} does not match n_classes {n_classes}'
            )

    wrges = []

    for k in range(n_classes):
        pred_full = prob_full[:, k]
        pred_reduced = prob_reduced[:, k]

        # WRGE uses same computation as WRGR
        wrge_k = wrgr_cramer(pred_full, pred_reduced)
        wrges.append(wrge_k)

        if verbose:
            print(f'Class {k}: WRGE = {wrge_k:.4f}')

    wrges = np.array(wrges)

    # Weighted average
    wrge_weighted = np.nansum(wrges * class_weights) / np.nansum(class_weights)

    return wrge_weighted, wrges, class_weights


def extract_features_from_images(images, feature_extractor, pca, scaler,
                                 device, batch_size=64):
    """
    Extract and transform features from images.

    Parameters
    ----------
    images : torch.Tensor
        Input images
    feature_extractor : torch.nn.Module
        Feature extraction model (e.g., ResNet)
    pca : sklearn PCA
        Fitted PCA transformer
    scaler : sklearn StandardScaler
        Fitted scaler
    device : torch.device
        Device for computation
    batch_size : int
        Batch size for processing

    Returns
    -------
    np.ndarray
        Transformed features ready for classifier
    """
    feature_extractor.eval()

    features_list = []

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].to(device)
            features = feature_extractor(batch).cpu().numpy()
            features_list.append(features)
            del batch, features
            gc.collect()

    features = np.vstack(features_list)
    del features_list

    # Apply PCA and scaling
    features_pca = pca.transform(features)
    del features
    features_scaled = scaler.transform(features_pca)
    del features_pca

    return features_scaled


def get_predictions_from_features(features, model, model_class_order,
                                  class_order, model_type='sklearn',
                                  device=None, batch_size=64):
    """
    Get predictions from features with proper class alignment.

    Parameters
    ----------
    features : np.ndarray
        Input features
    model : sklearn or PyTorch model
        Trained classifier
    model_class_order : array-like
        Order of classes in model's output
    class_order : array-like
        Target class order for alignment
    model_type : {'sklearn', 'pytorch'}
        Type of model
    device : torch.device, optional
        Device for PyTorch models
    batch_size : int
        Batch size for PyTorch models

    Returns
    -------
    np.ndarray
        Aligned probability predictions
    """
    if model_type == 'sklearn':
        probs = model.predict_proba(features)

    elif model_type == 'pytorch':
        model.eval()
        probs_list = []

        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch = features[i:i + batch_size]
                batch_tensor = torch.tensor(batch, dtype=torch.float32).to(device)
                logits = model(batch_tensor)
                probs_batch = torch.softmax(logits, dim=1).cpu().numpy()
                probs_list.append(probs_batch)
                del batch, batch_tensor, logits, probs_batch
                gc.collect()

        probs = np.vstack(probs_list)
        del probs_list

    else:
        raise ValueError(f"model_type must be 'sklearn' or 'pytorch', got '{model_type}'")

    # Align to target class order
    probs_aligned = align_proba_to_class_order(probs, model_class_order, class_order)
    del probs

    return probs_aligned



# Image Occlusion Function
def apply_patch_occlusion(images, num_patches, patch_size=32, random_seed=None):
    """
    Apply random patch occlusion to images.

    Parameters
    ----------
    images : torch.Tensor, shape (n_samples, channels, height, width)
        Input images
    num_patches : int
        Number of patches to occlude per image
    patch_size : int
        Size of square occlusion patches
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    torch.Tensor
        Images with occlusion applied
    """
    if random_seed is not None:
        random.seed(random_seed)
        torch.manual_seed(random_seed)

    images_occluded = images.clone()
    n_samples, channels, height, width = images.shape

    if num_patches > 0:
        for i in range(n_samples):
            for _ in range(num_patches):
                y0 = random.randint(0, height - patch_size)
                x0 = random.randint(0, width - patch_size)
                images_occluded[i, :, y0:y0 + patch_size, x0:x0 + patch_size] = 0.0

    return images_occluded


#WRGE Evaluation
def evaluate_wrge_multiclass_occlusion(model, feature_extractor, pca, scaler,
                                       images_dataset, removal_fractions,
                                       model_class_order, class_order,
                                       model_type='sklearn', device=None,
                                       patch_size=32, batch_size=64,
                                       class_weights=None, model_name='Model', wrga_full=None,
                                       plot=True, fig_size=(10, 6), verbose=True,
                                       random_seed=None):
    """
    Evaluate WRGE for multiclass classification with image occlusion.

    Parameters
    ----------
    model : sklearn or PyTorch model
        Trained classifier to evaluate
    feature_extractor : torch.nn.Module
        Feature extraction model (e.g., ResNet)
    pca : sklearn PCA
        Fitted PCA transformer
    scaler : sklearn StandardScaler
        Fitted scaler
    images_dataset : torch.utils.data.Dataset
        Images
    removal_fractions : array-like
        Fractions of pixels to remove (0.0 to 1.0)
    model_class_order : array-like
        Order of classes in model's output
    class_order : array-like
        Target class order for alignment
    model_type : {'sklearn', 'pytorch'}
        Type of model
    device : torch.device
        Device for computation
    patch_size : int
        Size of occlusion patches
    batch_size : int
        Batch size for processing
    class_weights : array-like, optional
        Custom weights for each class
    model_name : str
        Name of model for display
    wrga_full : float, optional
        Full WRGA score for rescaling. If None, no rescaling is applied.
    plot : bool
        Whether to generate visualization
    fig_size : tuple, optional
        Figure size for plot
    verbose : bool
        Print detailed results
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict
        Dictionary containing:
        - 'wrge_scores': WRGE scores at each removal fraction
        - 'aurge': Area under RGE curve
        - 'removal_fractions': Removal fractions tested
        - 'per_class_wrge': Per-class WRGE at each removal fraction
        - 'class_order': Class order used
    """
    removal_fractions = np.asarray(removal_fractions)

    if verbose:
        print(f'WRGE Evaluation: {model_name}')
        print(f'Testing {len(removal_fractions)} removal fractions')

    # Get image dimensions
    dataloader_temp = DataLoader(images_dataset, batch_size=1)
    sample_img = next(iter(dataloader_temp))[0]
    _, channels, height, width = sample_img.shape
    total_pixels = height * width
    patch_pixels = patch_size * patch_size
    del dataloader_temp, sample_img
    gc.collect()

    # Get original predictions
    if verbose:
        print('\nExtracting features from original images')

    dataloader = DataLoader(images_dataset, batch_size=batch_size, shuffle=False)
    images_list = []
    for batch_tuple in dataloader:
        batch = batch_tuple[0] if isinstance(batch_tuple, (list, tuple)) else batch_tuple
        images_list.append(batch)
    images_all = torch.cat(images_list, dim=0)
    del images_list, dataloader
    gc.collect()

    features_original = extract_features_from_images(
        images_all, feature_extractor, pca, scaler, device, batch_size
    )

    prob_original = get_predictions_from_features(
        features_original, model, model_class_order, class_order,
        model_type, device, batch_size
    )
    del features_original
    gc.collect()

    # Evaluate at each removal fraction
    wrge_scores = []
    per_class_wrge_list = []

    for frac in removal_fractions:
        if verbose:
            print(f'\nProcessing removal fraction: {frac * 100:.0f}%')

        pixels_to_remove = int(frac * total_pixels)
        num_patches = pixels_to_remove // patch_pixels

        # Apply occlusion
        images_occluded = apply_patch_occlusion(
            images_all, num_patches, patch_size, random_seed
        )

        # Extract features from occluded images
        features_occluded = extract_features_from_images(
            images_occluded, feature_extractor, pca, scaler, device, batch_size
        )
        del images_occluded
        gc.collect()

        # Get predictions
        prob_occluded = get_predictions_from_features(
            features_occluded, model, model_class_order, class_order,
            model_type, device, batch_size
        )
        del features_occluded
        gc.collect()

        # Calculate WRGE
        wrge_val, wrge_per_class, _ = wrge_cramer_multiclass(
            prob_original,
            prob_occluded,
            class_weights=class_weights,
            verbose=False
        )

        wrge_scores.append(0.0 if np.isnan(wrge_val) else wrge_val)
        per_class_wrge_list.append(wrge_per_class)

        if verbose:
            print(f'WRGE = {wrge_scores[-1]:.4f}')

        del prob_occluded
        gc.collect()

    wrge_scores = np.array(wrge_scores)
    per_class_wrge_list = np.array(per_class_wrge_list)

    # Normalize removal fractions for AUC
    max_frac = np.max(removal_fractions)
    removal_norm = removal_fractions / max_frac if max_frac > 0 else removal_fractions

    if wrga_full is not None and np.isfinite(wrga_full):
        wrge_rescaled = wrge_scores * float(wrga_full)
    else:
        wrge_rescaled = wrge_scores

    # Calculate AURGE
    aurge = auc(removal_norm, wrge_rescaled)

    if verbose:
        print(f"AURGE: {aurge:.4f}")

    # Visualization
    if plot:
        plt.figure(figsize=fig_size)
        plt.plot(removal_fractions * 100, wrge_scores, '-o', linewidth=2.5,
                 markersize=6, color='forestgreen',
                 label=f'{model_name} (AURGE={aurge:.3f})')
        plt.fill_between(removal_fractions * 100, 0, wrge_scores,
                         alpha=0.2, color='forestgreen')
        plt.xlabel('% Pixels Removed', fontsize=11, fontweight='bold')
        plt.ylabel('RGE Score', fontsize=11, fontweight='bold')
        plt.title(f'RGE Occlusion Curve: {model_name}', fontsize=12, fontweight='bold')
        plt.grid(alpha=0.3, linestyle='--')
        plt.xlim([0, removal_fractions[-1] * 100])
        max_score = max(wrge_scores) if len(wrge_scores) > 0 else 1
        plt.ylim([0, max_score * 1.1])
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()

    # Cleanup
    del images_all, prob_original
    gc.collect()

    return {
        'wrge_scores': wrge_scores,
        'aurge': aurge,
        'removal_fractions': removal_fractions,
        'per_class_wrge': per_class_wrge_list,
        'class_order': class_order
    }


def compare_models_wrge(models_dict, feature_extractor, pca, scaler,
                        images_dataset, removal_fractions, class_order,
                        patch_size=32, batch_size=64, class_weights=None,
                        device=None, fig_size=(12, 6), verbose=True,
                        random_seed=None):
    """
    Compare explainability of multiple models using WRGE metrics.

    Parameters
    ----------
    models_dict : dict
        Dictionary mapping model names to tuples of:
        (model, model_class_order, model_type)

        Example: {
            'RF': (rf_model, rf.classes_, 'sklearn'),
            'VQC': (vqc_model, np.array([0,1,2]), 'pytorch')
        }
    feature_extractor : torch.nn.Module
        Feature extraction model (shared across all models)
    pca : sklearn PCA
        Fitted PCA transformer (shared)
    scaler : sklearn StandardScaler
        Fitted scaler (shared)
    images_dataset : torch.utils.data.Dataset
        Images
    removal_fractions : array-like
        Fractions of pixels to remove
    class_order : array-like
        Target class order for alignment (shared)
    patch_size : int
        Size of occlusion patches
    batch_size : int
        Batch size for processing
    class_weights : array-like, optional
        Class weights for all models
    device : torch.device
        Device for computation
    fig_size : tuple, optional
        Figure size for comparison plot
    verbose : bool
        Print detailed results
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict
        WRGE evaluation results for all models
    """
    results = {}

    for model_name, (model, model_class_order, model_type) in models_dict.items():
        if verbose:
            print(f'\nEvaluating {model_name}')

        result = evaluate_wrge_multiclass_occlusion(
            model=model,
            feature_extractor=feature_extractor,
            pca=pca,
            scaler=scaler,
            images_dataset=images_dataset,
            removal_fractions=removal_fractions,
            model_class_order=model_class_order,
            class_order=class_order,
            model_type=model_type,
            device=device,
            patch_size=patch_size,
            batch_size=batch_size,
            class_weights=class_weights,
            model_name=model_name,
            plot=False,
            verbose=verbose,
            random_seed=random_seed
        )
        results[model_name] = result

    # Comparison plot
    plt.figure(figsize=fig_size)

    cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, len(results)))

    for (model_name, result), color in zip(results.items(), colors):
        plt.plot(
            result['removal_fractions'] * 100,
            result['wrge_scores'],
            '-o',
            linewidth=2.5,
            markersize=5,
            color=color,
            label=f"{model_name} (AURGE={result['aurge']:.3f})"
        )

    plt.xlabel('% Pixels Removed', fontsize=11, fontweight='bold')
    plt.ylabel('WRGE Score', fontsize=11, fontweight='bold')
    plt.title('WRGE Occlusion Comparison', fontsize=12, fontweight='bold')
    plt.grid(alpha=0.3, linestyle='--')
    plt.xlim([0, removal_fractions[-1] * 100])
    plt.legend(fontsize=9)

    plt.tight_layout()
    plt.show()

    # Print comparison summary
    if verbose:
        print("Explainability Comparison Summary (AURGE)")

        model_names = list(results.keys())
        aurge_scores = np.array([results[name]['aurge'] for name in model_names])

        for name, score in zip(model_names, aurge_scores):
            print(f"{name}: AURGE = {score:.4f}")

        if len(model_names) >= 2:
            best_idx = int(np.nanargmax(aurge_scores))
            worst_idx = int(np.nanargmin(aurge_scores))

            print(f"Best: {model_names[best_idx]} (AURGE={aurge_scores[best_idx]:.4f})")
            print(f"Worst: {model_names[worst_idx]} (AURGE={aurge_scores[worst_idx]:.4f})")

    return results

