import matplotlib.pyplot as plt
import numpy as np
import os
import re
from sklearn.metrics import auc

from safeai_files.check_explainability import compute_rge_values
from safeai_files.check_robustness import rgr_all
from safeai_files.core import partial_rga_with_curves, rga


def safeai_values(x_train, x_test, y_test, y_prob, model, data_name, save_path):
    """
    Compute SafeAI lists of values and plot curves: Accuracy (RGA), Explainability (RGE AUC), Robustness (RGR AUC).

    Parameters:
    -------------
    x_train: pandas.DataFrame
        Training data features.
    x_test: pandas.DataFrame
        Test data features.
    y_test: pd.DataFrame
        True labels for test data.
    y_prob: list
        Predicted probabilities for the positive class.
    model: Union[CatBoostClassifier, CatBoostRegressor, XGBClassifier, XGBRegressor, BaseEstimator, torch.nn.Module]
        Trained classifier used in compute_rge_values and rgr_all.
    data_name: str
        Name of the dataset to show on the graph
    save_path: str
        Directory for saving graphs

    Returns:
    --------
    dict containing:
        rga_value: float
        rge_auc: float
        rgr_auc: float
        x_final: list of float
        y_final: list of float
        z_final: list of float
    """

    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Accuracy (RGA)
    rga_value = rga(y_test, y_prob)

    # Explainability (RGE)
    explain = x_train.columns.tolist()
    remaining_vars = explain.copy()
    removed_vars = []
    step_rges = []

    for k in range(0, len(explain) + 1):
        if k == 0:
            step_rges.append(1.0)
            continue

        candidate_rges = []
        for var in remaining_vars:
            current_vars = removed_vars + [var]
            rge_k = compute_rge_values(x_train, x_test, y_prob, model, current_vars, group=True)
            candidate_rges.append((var, rge_k.iloc[0, 0]))

        best_var, best_rge = max(candidate_rges, key=lambda x: x[1])
        removed_vars.append(best_var)
        remaining_vars.remove(best_var)
        step_rges.append(best_rge)

    x_rge = np.linspace(0, 1, len(step_rges))
    y_rge = np.array(step_rges)
    rge_auc = auc(x_rge, y_rge)

    # Plot
    model_name = model.__class__.__name__
    model_name_spaced = ' '.join(re.findall(r'[A-Z]{2,}(?=[A-Z][a-z]|[A-Z]*$)|[A-Z][a-z]*', model_name))

    plt.figure(figsize=(6, 4))
    plt.plot(x_rge, y_rge, marker='o', label=f"RGE Curve (AURGE = {rge_auc:.4f})")
    # Plot baseline only if not a Dummy model
    if model_name not in ["DummyRegressor", "DummyClassifier"]:
        random_baseline = float(y_rge[-1])
        plt.axhline(random_baseline, color='red', linestyle='--',
                    label=f"Random Baseline (RGE = {random_baseline:.2f})")
    plt.xlabel("Fraction of Variables Removed")
    plt.ylabel("RGE")
    plt.title(f"{model_name_spaced} RGE Curve ({data_name})")
    plt.legend()
    plt.grid(True)

    # Save the plot
    model_name_clean = model_name_spaced.lower().replace(" ", "_")
    data_name_clean = data_name.lower().replace(" ", "_")
    filename_rge = f"{model_name_clean}_rge_{data_name_clean}.png"

    full_path_rge = os.path.join(save_path, filename_rge)
    plt.savefig(full_path_rge, dpi=300)
    plt.close()

    # Robustness (RGR)
    thresholds = np.arange(0, 0.51, 0.01)
    rgr_scores = [rgr_all(x_test, y_prob, model, t, 'gaussian_noise', seed=78925) for t in thresholds]
    normalized_t = thresholds / 0.5
    rgr_auc = auc(normalized_t, rgr_scores)

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(normalized_t, rgr_scores, linestyle='-', label=f"RGR Curve (AURGR = {rgr_auc:.4f})")
    plt.title(f'{model_name_spaced} RGR Curve ({data_name})')
    if model_name not in ["DummyRegressor", "DummyClassifier"]:
        plt.axhline(0.5, color='red', linestyle='--', label=f"Random Baseline (RGR = 0.5)")
    plt.xlabel('Normalized Perturbation')
    plt.ylabel('RGR')
    plt.legend()
    plt.xlim([0, 1])
    plt.grid(True)

    # Save the plot
    filename_rgr = f"{model_name_clean}_rgr_{data_name_clean}.png"
    full_path_rgr = os.path.join(save_path, filename_rgr)
    plt.savefig(full_path_rgr, dpi=300)
    plt.close()

    # Values for final compliance score
    # RGA
    num_steps = len(step_rges) - 1
    step_rgas = []
    thresholds_rga = np.linspace(1, 0, num_steps + 1)
    for i in range(num_steps):
        lower = float(thresholds_rga[i + 1])
        upper = float(thresholds_rga[i])
        partial = partial_rga_with_curves(y_test, y_prob, lower, upper, False)
        step_rgas.append(partial)
    reverse_cumulative = np.cumsum(step_rgas[::-1])[::-1]
    x_final = np.concatenate((reverse_cumulative, [0.])).tolist()

    # RGE
    y_final = step_rges

    # RGR
    num_steps_rgr = len(step_rges)
    thresholds_rgr = np.linspace(0, 0.5, num_steps_rgr)
    z_final = [rgr_all(x_test, y_prob, model, t, 'gaussian_noise', seed=78925) for t in thresholds_rgr]

    return {
        'model_name': model.__class__.__name__,
        'rga_value': rga_value,
        'rge_auc': rge_auc,
        'rgr_auc': rgr_auc,
        'x_final': x_final,
        'y_final': y_final,
        'z_final': z_final,
        'x_rge': x_rge.tolist(),
        'y_rge': y_rge.tolist(),
        'x_rgr': normalized_t.tolist(),
        'y_rgr': rgr_scores
    }



