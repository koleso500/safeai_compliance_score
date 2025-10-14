import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from src.config import DATASET_CONFIG, MODELS_TO_TRAIN, PATHS, TOPSIS_IDEAL_VALUES, TOPSIS_WORST_VALUES
from safeai_files.utils import plot_mean_histogram, plot_model_curves, plot_diff_mean_histogram


def load_safeai_results(model_name, dataset_name):
    """
    Load SafeAI results for a model

    Parameters
    ----------
    model_name: Name of the model
    dataset_name: Name of the dataset

    Returns
    -------
    Dictionary with x_final, y_final, z_final or None

    """

    file_path = os.path.join(
        PATHS["results_dir"],
        f"{model_name}_{dataset_name}_evaluation.json"
        )

    if not os.path.exists(file_path):
        print(f"Warning: Evaluation file not found for {model_name}")
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

            # Extract compliance data
            compliance = data.get('compliance', {})
            return {
                'x_final': compliance.get('x_final'),
                'y_final': compliance.get('y_final'),
                'z_final': compliance.get('z_final'),
            }
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def extract_safeai_values(safeai_data):
    """
    Extract x_final, y_final, z_final from SafeAI results

    Parameters
    ----------
    safeai_data: Dictionary with SafeAI metrics

    Returns
    -------
    Tuple of (x, y, z) or (None, None, None) if not found

    """

    if safeai_data is None:
        return None, None, None

    x = safeai_data.get('x_final', None)
    y = safeai_data.get('y_final', None)
    z = safeai_data.get('z_final', None)

    return x, y, z


def load_all_model_results(models, dataset_name):
    """
    Load SafeAI results for all models

    Parameters
    ----------
    models: List of model names
    dataset_name: Name of the dataset

    Returns
    -------
    Dictionary {model_name: (x, y, z)}

    """

    all_results = {}

    for model_name in models:
        safeai_data = load_safeai_results(model_name, dataset_name)
        x, y, z = extract_safeai_values(safeai_data)

        if x is not None and y is not None and z is not None:
            all_results[model_name] = {
                'x': x,
                'y': y,
                'z': z,
                'data': safeai_data
            }
        else:
            print(f"{model_name}: Incomplete SafeAI data (missing x, y, or z)")

    return all_results


def calculate_differences(models_results, baseline_model='random'):
    """
    Calculate performance differences relative to baseline

    Parameters
    ----------
    models_results: Dictionary of all model results
    baseline_model: Name of baseline model

    Returns
    -------
    Dictionary with difference values

    """

    differences = {}

    if baseline_model not in models_results:
        print(f"Warning: Baseline model '{baseline_model}' not found")
        return differences

    if models_results[baseline_model]['x'] is None:
        print(f"Warning: Baseline model '{baseline_model}' has no valid data")
        return differences

    baseline_x = np.array(models_results[baseline_model]['x'])
    baseline_y = np.array(models_results[baseline_model]['y'])
    baseline_z = np.array(models_results[baseline_model]['z'])

    for model_name, result in models_results.items():
        if model_name == baseline_model:
            continue

        x_diff = (np.array(result['x']) - baseline_x).tolist()
        y_diff = (np.array(result['y']) - baseline_y).tolist()
        z_diff = (np.array(result['z']) - baseline_z).tolist()

        differences[model_name] = {
            'x': x_diff,
            'y': y_diff,
            'z': z_diff
        }

    return differences


def plot_all_model_curves(models_results, dataset_name):
    """
    Plot all model curves

    Parameters
    ----------
    models_results: Dictionary of all model results
    dataset_name: Name of the dataset

    Returns
    -------

    """

    num_models = len(models_results)
    cols = 2
    rows = (num_models + 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axs = axs.flatten()

    x_rga = np.linspace(0, 1, len(next(iter(models_results.values()))['y']))

    for idx, (model_name, result) in enumerate(models_results.items()):
        plot_model_curves(
            x_rga,
            [result['x'], result['y'], result['z']],
            model_name=model_name,
            title=f"{model_name} Curves ({dataset_name})",
            ax=axs[idx]
        )

    # Hide unused subplots
    for idx in range(num_models, len(axs)):
        axs[idx].set_visible(False)

    fig.suptitle(f'All Model Curves - {dataset_name}', fontsize=14, fontweight='bold')
    fig.subplots_adjust(top=0.95, hspace=0.4)
    plt.tight_layout()
    plt.show()


def plot_difference_curves(models_results, differences, dataset_name):
    """
    Plot performance differences relative to baseline

    Parameters
    ----------
    models_results: Dictionary of all model results
    differences: Dictionary of difference values
    dataset_name: Name of the dataset

    Returns
    -------

    """

    num_models = len(differences)
    cols = 2
    rows = (num_models + 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axs = axs.flatten()

    x_rga = np.linspace(0, 1, len(next(iter(models_results.values()))['y']))

    for idx, (model_name, diff) in enumerate(differences.items()):
        plot_model_curves(
            x_rga,
            [diff['x'], diff['y'], diff['z']],
            model_name="Baseline",
            prefix="Difference",
            title=f"{model_name} Difference vs Baseline ({dataset_name})",
            ax=axs[idx]
        )

    # Hide unused subplots
    for idx in range(num_models, len(axs)):
        axs[idx].set_visible(False)

    fig.suptitle(f'Performance Differences vs Baseline - {dataset_name}', fontsize=14, fontweight='bold')
    fig.subplots_adjust(top=0.95, hspace=0.4)
    plt.tight_layout()
    plt.show()


def plot_mean_histograms(models_results, dataset_name, mean_type="arithmetic"):
    """
    Plot mean histograms for all models

    Parameters
    ----------
    models_results: Dictionary of all model results
    dataset_name: Name of the dataset
    mean_type: "arithmetic", "geometric", or "quadratic"

    Returns
    -------

    """

    num_models = len(models_results)
    cols = 2
    rows = (num_models + 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axs = axs.flatten()

    for idx, (model_name, result) in enumerate(models_results.items()):
        # Create meshgrid
        x_array = np.array(result['x'])
        y_array = np.array(result['y'])
        z_array = np.array(result['z'])

        x_grid, y_grid, z_grid = np.meshgrid(x_array, y_array, z_array, indexing='ij')

        plot_mean_histogram(
            x_grid, y_grid, z_grid,
            model_name=model_name,
            bar_label=model_name,
            mean_type=mean_type,
            ax=axs[idx]
        )

    # Hide unused subplots
    for idx in range(num_models, len(axs)):
        axs[idx].set_visible(False)

    fig.suptitle(f'{mean_type.capitalize()} Mean Histograms - {dataset_name}',
                 fontsize=14, fontweight='bold')
    fig.subplots_adjust(top=0.95, hspace=0.4)
    plt.tight_layout()

    # Save plots
    plots_dir = os.path.join(PATHS["results_dir"], "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f"mean_histograms_{mean_type}_{dataset_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {plot_path}")
    plt.close()


def plot_diff_mean_histograms(differences, dataset_name, mean_type="arithmetic"):
    """
    Plot mean histograms for performance differences vs baseline

    Parameters
    ----------
    differences: Dictionary of difference values
    dataset_name: Name of the dataset
    mean_type: "arithmetic", "geometric", or "quadratic"

    Returns
    -------

    """

    if not differences:
        print(f"No difference data available, skipping {mean_type} difference histograms")
        return

    num_models = len(differences)
    cols = 2
    rows = (num_models + 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axs = axs.flatten()

    for idx, (model_name, diff) in enumerate(differences.items()):
        # Create meshgrid
        x_array = np.array(diff['x'])
        y_array = np.array(diff['y'])
        z_array = np.array(diff['z'])

        x_grid, y_grid, z_grid = np.meshgrid(x_array, y_array, z_array, indexing='ij')

        plot_diff_mean_histogram(
            x_grid, y_grid, z_grid,
            model_name=model_name,
            bar_label=model_name,
            mean_type=mean_type,
            ax=axs[idx]
        )

    # Hide unused subplots
    for idx in range(num_models, len(axs)):
        axs[idx].set_visible(False)

    fig.suptitle(f'{mean_type.capitalize()} Mean Differences vs Baseline - {dataset_name}',
                 fontsize=14, fontweight='bold')
    fig.subplots_adjust(top=0.95, hspace=0.4)
    plt.tight_layout()

    # Save plots
    plots_dir = os.path.join(PATHS["results_dir"], "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f"differences_mean_histograms_{mean_type}_{dataset_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {plot_path}")
    plt.close()


def calculate_topsis_ranking(models_results, ideal_values: dict, worst_values: dict, weights=None):
    """

    Parameters
    ----------
    models_results: Dictionary of all model results
    weights: Weights for [x, y, z] metrics (default: equal weights)
    ideal_values: Dict with 'x', 'y', 'z' lists of ideal values
    worst_values: Dict with 'x', 'y', 'z' lists of worst values

    Returns
    -------
    DataFrame with TOPSIS scores and rankings

    """

    if weights is None:
        weights = np.array([1 / 3, 1 / 3, 1 / 3])

    # Calculate means
    means = {}
    for model_name, result in models_results.items():
        means[model_name] = {
            'mean_x': np.mean(result['x']),
            'mean_y': np.mean(result['y']),
            'mean_z': np.mean(result['z']),
        }

    # Create DataFrame
    df = pd.DataFrame.from_dict(means, orient='index')

    # Normalize
    for col in df.columns:
        vec = df[col].values.astype(float)
        norm = np.sqrt((vec ** 2).sum())
        df[f'r_{col}'] = vec / norm if norm > 0 else 0

    # Apply weights
    df['v_mean_x'] = df['r_mean_x'] * weights[0]
    df['v_mean_y'] = df['r_mean_y'] * weights[1]
    df['v_mean_z'] = df['r_mean_z'] * weights[2]

    # Normalize ideal and worst values using norms from data
    norm_x = np.sqrt((df["mean_x"].values ** 2).sum())
    norm_y = np.sqrt((df["mean_y"].values ** 2).sum())
    norm_z = np.sqrt((df["mean_z"].values ** 2).sum())

    x_plus = np.mean(ideal_values['x']) / norm_x if norm_x > 0 else 0
    x_minus = np.mean(worst_values['x']) / norm_x if norm_x > 0 else 0

    y_plus = np.mean(ideal_values['y']) / norm_y if norm_y > 0 else 0
    y_minus = np.mean(worst_values['y']) / norm_y if norm_y > 0 else 0

    z_plus = np.mean(ideal_values['z']) / norm_z if norm_z > 0 else 0
    z_minus = np.mean(worst_values['z']) / norm_z if norm_z > 0 else 0

    v_x_plus = x_plus * weights[0]
    v_x_minus = x_minus * weights[0]

    v_y_plus = y_plus * weights[1]
    v_y_minus = y_minus * weights[1]

    v_z_plus = z_plus * weights[2]
    v_z_minus = z_minus * weights[2]

    # Calculate distances
    df['S_plus'] = np.sqrt(
        (df['v_mean_x'] - v_x_plus) ** 2 +
        (df['v_mean_y'] - v_y_plus) ** 2 +
        (df['v_mean_z'] - v_z_plus) ** 2
    )

    df['S_minus'] = np.sqrt(
        (df['v_mean_x'] - v_x_minus) ** 2 +
        (df['v_mean_y'] - v_y_minus) ** 2 +
        (df['v_mean_z'] - v_z_minus) ** 2
    )

    # TOPSIS score
    df['C'] = df.apply(
        lambda row: row['S_minus'] / (row['S_plus'] + row['S_minus'])
        if (row['S_plus'] + row['S_minus']) > 0 else 0.5,
        axis=1
    )

    df['Rank'] = df['C'].rank(ascending=False, method='min').astype(int)

    return df.sort_values('C', ascending=False)


def run_safeai_compliance_score():
    """
    Main pipeline for SafeAI Compliance Score
    """
    print("\n" + "=" * 60)
    print("SafeAI METRICS VISUALIZATION PIPELINE")
    print("=" * 60)
    print(f"Dataset: {DATASET_CONFIG['dataset_name']}")
    print(f"Models: {MODELS_TO_TRAIN}")
    print("=" * 60)

    # Load all model results
    print("\nLoading SafeAI results...")
    models_results = load_all_model_results(MODELS_TO_TRAIN, DATASET_CONFIG['dataset_name'])

    if not models_results:
        print("No SafeAI results found!")
        return

    print(f"Loaded {len(models_results)} models")

    # Plot all curves
    print("\nPlotting all model curves...")
    plot_all_model_curves(models_results, DATASET_CONFIG['dataset_name'])

    # Calculate and plot differences
    print("\nCalculating performance differences...")
    differences = calculate_differences(models_results, baseline_model='random')

    if differences:
        print("Plotting difference curves...")
        plot_difference_curves(models_results, differences, DATASET_CONFIG['dataset_name'])

    # Plot histograms
    for mean_type in ['arithmetic', 'geometric', 'quadratic']:
        print(f"\nPlotting {mean_type} mean histograms...")
        plot_mean_histograms(models_results, DATASET_CONFIG['dataset_name'], mean_type)

        print(f"Plotting {mean_type} mean difference histograms...")
        plot_diff_mean_histograms(differences, DATASET_CONFIG['dataset_name'], mean_type)

    # TOPSIS Ranking
    print("\n" + "=" * 60)
    print("TOPSIS RANKING")
    print("=" * 60)

    ranking_df = calculate_topsis_ranking(
        models_results,
        ideal_values=TOPSIS_IDEAL_VALUES,
        worst_values=TOPSIS_WORST_VALUES
    )
    print("\n", ranking_df[['mean_x', 'mean_y', 'mean_z', 'C', 'Rank']])

    # Save ranking
    ranking_path = os.path.join(
        PATHS["results_dir"],
        f"safeai_compliance_ranking_{DATASET_CONFIG['dataset_name']}.csv"
    )
    ranking_df.to_csv(ranking_path)
    print(f"\nRanking saved: {ranking_path}")

    return models_results, ranking_df


if __name__ == "__main__":
    models_results, ranking = run_safeai_compliance_score()