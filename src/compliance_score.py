import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from matplotlib.axes import Axes

from src.config import DATASET_CONFIG, MODELS_TO_TRAIN, PATHS, TOPSIS_IDEAL_VALUES, TOPSIS_WORST_VALUES
from safeai_files.utils import plot_mean_histogram, plot_model_curves, plot_diff_mean_histogram

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_safeai_results(model_name, dataset_name):
    """
    Load SafeAI results for a model

    Parameters
    ----------
    model_name: Name of the model
    dataset_name: Name of the dataset

    Returns
    -------
    Dictionary with x_final, y_final, z_final or None if not found/invalid

    """
    file_path = os.path.join(
        PATHS['results_dir'],
        f'{model_name}_{dataset_name}_evaluation.json'
    )

    if not os.path.exists(file_path):
        logger.warning('Evaluation file not found for %s: %s', model_name, file_path)
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

            # Extract compliance data
            compliance = data.get('compliance', None)

            if compliance is None:
                logger.warning("No 'compliance' key found in evaluation file for %s", model_name)
                return None

            return {
                'x_final': compliance.get('x_final'),
                'y_final': compliance.get('y_final'),
                'z_final': compliance.get('z_final'),
            }

    except json.JSONDecodeError as json_error:
        logger.error('Invalid JSON in %s: %s', file_path, json_error)
    except Exception as load_error:
        logger.error('Error loading SafeAI results from %s: %s', file_path, load_error)

    return None


def extract_safeai_values(safeai_data):
    """
    Extract x_final, y_final, z_final from SAFE-AI results with validation

    Parameters
    ----------
    safeai_data: Dictionary with SAFE-AI metrics

    Returns
    -------
    Tuple of (x, y, z) or (None, None, None) if not found or invalid

    """
    if safeai_data is None:
        return None, None, None

    x = safeai_data.get('x_final')
    y = safeai_data.get('y_final')
    z = safeai_data.get('z_final')

    # Validate that values are lists/arrays and not empty
    def validate_array(name: str, arr):
        if arr is None:
            return None
        if not isinstance(arr, (list, np.ndarray)):
            logger.warning('%s is not a list/array: %s', name, type(arr))
            return None
        if len(arr) == 0:
            logger.warning('%s is empty', name)
            return None
        return list(arr)

    x_valid = validate_array('x_final', x)
    y_valid = validate_array('y_final', y)
    z_valid = validate_array('z_final', z)

    return x_valid, y_valid, z_valid


def load_all_model_results(models, dataset_name):
    """
    Load SafeAI results for all models with validation

    Parameters
    ----------
    models: List of model names
    dataset_name: Name of the dataset

    Returns
    -------
    Dictionary {model_name: {'x': x, 'y': y, 'z': z, 'data': safeai_data}}

    """
    all_results = {}
    failed_models = []

    for model_name in models:
        try:
            safeai_data = load_safeai_results(model_name, dataset_name)
            x, y, z = extract_safeai_values(safeai_data)

            if x is not None and y is not None and z is not None:
                # Validate array lengths match
                if len(x) == len(y) == len(z):
                    all_results[model_name] = {
                        'x': x,
                        'y': y,
                        'z': z,
                        'data': safeai_data
                    }
                    logger.info('Loaded SafeAI data for %s', model_name)
                else:
                    logger.warning(
                        '%s: Mismatched array lengths (x=%d, y=%d, z=%d)',
                        model_name,
                        len(x),
                        len(y),
                        len(z),
                    )
                    failed_models.append(model_name)
            else:
                logger.warning('%s: Incomplete SafeAI data (missing x,y or z)', model_name)
                failed_models.append(model_name)

        except Exception as model_error:
            logger.error('Failed to load SafeAI results for %s: %s', model_name, model_error)
            failed_models.append(model_name)

    if failed_models:
        logger.warning('Failed to load SafeAI results for %d model(s): %s', len(failed_models), failed_models)

    return all_results


def calculate_differences(model_results, baseline_model='random'):
    """
    Calculate performance differences relative to baseline

    Parameters
    ----------
    model_results: Dictionary of all model results
    baseline_model: Name of baseline model (default: 'random')

    Returns
    -------
    Dictionary with difference values, or empty dict if baseline not found

    """

    differences = {}

    if baseline_model not in model_results:
        logger.warning("Baseline model '%s' not found among results.", baseline_model)
        return differences

    baseline = model_results[baseline_model]

    if any(baseline.get(k) is None for k in ("x", "y", "z")):
        logger.warning("Baseline model '%s' has invalid data.", baseline_model)
        return differences

    try:
        baseline_x = np.array(baseline['x'])
        baseline_y = np.array(baseline['y'])
        baseline_z = np.array(baseline['z'])

        for model_name, result in model_results.items():
            if model_name == baseline_model:
                continue

            try:
                model_x = np.array(result['x'])
                model_y = np.array(result['y'])
                model_z = np.array(result['z'])

                # Validate shapes match
                if (model_x.shape != baseline_x.shape or
                        model_y.shape != baseline_y.shape or
                        model_z.shape != baseline_z.shape
                ):
                    logger.warning(
                        'Shape mismatch for %s (x:%s vs %s); skipping difference calculation.',
                        model_name,
                        model_x.shape,
                        baseline_x.shape,
                    )
                    continue

                x_diff = (model_x - baseline_x).tolist()
                y_diff = (model_y - baseline_y).tolist()
                z_diff = (model_z - baseline_z).tolist()

                differences[model_name] = {
                    'x': x_diff,
                    'y': y_diff,
                    'z': z_diff
                }

            except Exception as diff_error:
                logger.error('Failed to calculate differences for %s: %s', model_name, diff_error)
                continue

    except Exception as baseline_error:
        logger.error("Failed to process baseline model '%s': %s", baseline_model, baseline_error)
        return {}

    return differences


def plot_all_model_curves(m_results, dataset_name):
    """
    Plot all model curves with error handling and robust subplot management.

    Parameters
    ----------
    m_results : dict
        Dictionary of all model results where keys are model names and values
        contain 'x', 'y', and 'z' arrays
    dataset_name : str
        Name of the dataset for labeling

    Returns
    -------
    str or None
        Path to saved plot if successful, None otherwise
    """
    if not m_results:
        logger.warning('No model results available for plotting curves')
        return None

    fig = None
    try:
        num_models = len(m_results)
        cols = 2
        rows = (num_models + cols - 1) // cols  # Ceiling division

        fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))

        # Flatten axes array for consistent indexing
        if num_models == 1:
            axs_flat = [axs]
        elif rows == 1 or cols == 1:
            axs_flat = list(axs.flat) if isinstance(axs, np.ndarray) else [axs]
        else:
            axs_flat = list(axs.flat)

        # Get x_range from first model
        first_result = next(iter(m_results.values()))
        x_range = np.linspace(0, 1, len(first_result['y']))

        # Plot each model
        for idx, (model_name, result) in enumerate(m_results.items()):
            ax: Axes = axs_flat[idx]
            try:
                plot_model_curves(
                    x_range,
                    [result['x'], result['y'], result['z']],
                    model_name=model_name,
                    title=f'{model_name} Curves ({dataset_name})',
                    ax=ax
                )
            except Exception as plot_error:
                logger.error(
                    'Failed to plot curves for %s: %s',
                    model_name,
                    plot_error,
                    exc_info=True
                )
                ax.text(
                    0.5, 0.5,
                    f"Error plotting {model_name}",
                    ha='center',
                    va='center',
                    transform=ax.transAxes,
                    fontsize=12,
                    color='red'
                )

        # Hide unused subplots
        for idx in range(num_models, len(axs_flat)):
            unused_ax: Axes = axs_flat[idx]
            unused_ax.axis('off')

        # Adjust layout and title
        fig.suptitle(
            f'All Model Curves - {dataset_name}',
            fontsize=16,
            fontweight='bold',
            y=0.995
        )
        plt.tight_layout(rect=(0, 0, 1, 0.99))

        # Save plot
        plots_dir = os.path.join(PATHS['results_dir'], 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        plot_filename = f'all_model_curves_{dataset_name}.png'
        plot_path = os.path.join(plots_dir, plot_filename)

        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info('All-model curves plot saved to: %s', plot_path)

        return plot_path

    except Exception as fig_error:
        logger.error(
            'Failed to create all-model curves plot: %s',
            fig_error,
            exc_info=True
        )
        return None
    finally:
        if fig is not None:
            plt.close(fig)


def plot_difference_curves(res, differences, dataset_name):
    """
    Plot performance differences relative to baseline with error handling.

    Parameters
    ----------
    res : dict
        Dictionary of all model results where keys are model names and values
        contain 'x', 'y', and 'z' arrays
    differences : dict
        Dictionary of difference values for each model
    dataset_name : str
        Name of the dataset for labeling

    Returns
    -------
    str or None
        Path to saved plot if successful, None otherwise
    """
    if not differences:
        logger.warning("No difference data to plot")
        return None

    fig = None
    try:
        num_models = len(differences)
        cols = 2
        rows = (num_models + cols - 1) // cols  # Ceiling division

        fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))

        # Flatten axes array for consistent indexing
        if num_models == 1:
            axs_flat = [axs]
        elif rows == 1 or cols == 1:
            axs_flat = list(axs.flat) if isinstance(axs, np.ndarray) else [axs]
        else:
            axs_flat = list(axs.flat)

        # Get x_range from first model
        first_result = next(iter(res.values()))
        x_range = np.linspace(0, 1, len(first_result['y']))

        # Plot each model's difference
        for idx, (model_name, diff) in enumerate(differences.items()):
            ax: Axes = axs_flat[idx]
            try:
                plot_model_curves(
                    x_range,
                    [diff['x'], diff['y'], diff['z']],
                    model_name="Baseline",
                    prefix="Difference",
                    title=f"{model_name} Difference vs Baseline ({dataset_name})",
                    ax=ax
                )
            except Exception as plot_error:
                logger.error(
                    'Failed to plot difference for %s: %s',
                    model_name,
                    plot_error,
                    exc_info=True
                )
                ax.text(
                    0.5, 0.5,
                    f"Error plotting {model_name}",
                    ha='center',
                    va='center',
                    transform=ax.transAxes,
                    fontsize=12,
                    color='red'
                )

        # Hide unused subplots
        for idx in range(num_models, len(axs_flat)):
            unused_ax: Axes = axs_flat[idx]
            unused_ax.axis('off')

        # Adjust layout and title
        fig.suptitle(
            f'Performance Differences vs Baseline - {dataset_name}',
            fontsize=16,
            fontweight='bold',
            y=0.995
        )
        plt.tight_layout(rect=(0, 0, 1, 0.99))

        # Save plot
        plots_dir = os.path.join(PATHS["results_dir"], "plots")
        os.makedirs(plots_dir, exist_ok=True)

        plot_filename = f"difference_curves_{dataset_name}.png"
        plot_path = os.path.join(plots_dir, plot_filename)

        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info('Difference curves plot saved to: %s', plot_path)

        return plot_path

    except Exception as fig_error:
        logger.error(
            'Failed to create difference curves plot: %s',
            fig_error,
            exc_info=True
        )
        return None
    finally:
        if fig is not None:
            plt.close(fig)


def plot_mean_histograms(values, dataset_name, mean_type="arithmetic"):
    """
    Plot mean histograms for all models with error handling.

    Parameters
    ----------
    values : dict
        Dictionary of all model results where keys are model names and values
        contain 'x', 'y', and 'z' arrays
    dataset_name : str
        Name of the dataset for labeling
    mean_type : str, optional
        Type of mean to calculate: "arithmetic", "geometric", or "quadratic"
        Default is "arithmetic"

    Returns
    -------
    str or None
        Path to saved plot if successful, None otherwise
    """
    if not values:
        logger.warning("No model results for histograms")
        return None

    fig = None
    try:
        num_models = len(values)
        cols = 2
        rows = (num_models + cols - 1) // cols  # Ceiling division

        fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))

        # Flatten axes array for consistent indexing
        if num_models == 1:
            axs_flat = [axs]
        elif rows == 1 or cols == 1:
            axs_flat = list(axs.flat) if isinstance(axs, np.ndarray) else [axs]
        else:
            axs_flat = list(axs.flat)

        for idx, (model_name, result) in enumerate(values.items()):
            ax: Axes = axs_flat[idx]
            try:
                # Create meshgrid
                x_array = np.array(result['x'])
                y_array = np.array(result['y'])
                z_array = np.array(result['z'])

                # Validate arrays are not empty
                if x_array.size == 0 or y_array.size == 0 or z_array.size == 0:
                    logger.warning('Empty arrays for %s, skipping histogram', model_name)
                    ax.text(
                        0.5, 0.5,
                        f"No data for {model_name}",
                        ha='center',
                        va='center',
                        transform=ax.transAxes,
                        fontsize=12,
                        color='orange'
                    )
                    continue

                x_grid, y_grid, z_grid = np.meshgrid(x_array, y_array, z_array, indexing='ij')

                plot_mean_histogram(
                    x_grid, y_grid, z_grid,
                    model_name=model_name,
                    bar_label=model_name,
                    mean_type=mean_type,
                    ax=ax
                )
            except Exception as plot_error:
                logger.error(
                    'Failed to plot histogram for %s: %s',
                    model_name,
                    plot_error,
                    exc_info=True
                )
                ax.text(
                    0.5, 0.5,
                    f"Error plotting {model_name}",
                    ha='center',
                    va='center',
                    transform=ax.transAxes,
                    fontsize=12,
                    color='red'
                )

        # Hide unused subplots
        for idx in range(num_models, len(axs_flat)):
            unused_ax: Axes = axs_flat[idx]
            unused_ax.axis('off')

        # Adjust layout and title
        fig.suptitle(
            f'{mean_type.capitalize()} Mean Histograms - {dataset_name}',
            fontsize=16,
            fontweight='bold',
            y=0.995
        )
        plt.tight_layout(rect=(0, 0, 1, 0.99))

        # Save plots
        plots_dir = os.path.join(PATHS["results_dir"], "plots")
        os.makedirs(plots_dir, exist_ok=True)

        plot_filename = f"mean_histograms_{mean_type}_{dataset_name}.png"
        plot_path = os.path.join(plots_dir, plot_filename)

        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info('Mean histograms plot saved to: %s', plot_path)

        return plot_path

    except Exception as fig_error:
        logger.error(
            'Failed to create %s histograms: %s',
            mean_type,
            fig_error,
            exc_info=True
        )
        return None
    finally:
        if fig is not None:
            plt.close(fig)


def plot_diff_mean_histograms(differences, dataset_name, mean_type="arithmetic"):
    """
    Plot mean histograms for performance differences vs baseline.

    Parameters
    ----------
    differences : dict
        Dictionary of difference values for each model
    dataset_name : str
        Name of the dataset for labeling
    mean_type : str, optional
        Type of mean to calculate: "arithmetic", "geometric", or "quadratic"
        Default is "arithmetic"

    Returns
    -------
    str or None
        Path to saved plot if successful, None otherwise
    """
    if not differences:
        logger.info('No difference data available, skipping %s difference histograms', mean_type)
        return None

    fig = None
    try:
        num_models = len(differences)
        cols = 2
        rows = (num_models + cols - 1) // cols  # Ceiling division

        fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))

        # Flatten axes array for consistent indexing
        if num_models == 1:
            axs_flat = [axs]
        elif rows == 1 or cols == 1:
            axs_flat = list(axs.flat) if isinstance(axs, np.ndarray) else [axs]
        else:
            axs_flat = list(axs.flat)

        for idx, (model_name, diff) in enumerate(differences.items()):
            ax: Axes = axs_flat[idx]
            try:
                # Create meshgrid
                x_array = np.array(diff['x'])
                y_array = np.array(diff['y'])
                z_array = np.array(diff['z'])

                if x_array.size == 0 or y_array.size == 0 or z_array.size == 0:
                    logger.warning('Empty difference arrays for %s', model_name)
                    ax.text(
                        0.5, 0.5,
                        f"No data for {model_name}",
                        ha='center',
                        va='center',
                        transform=ax.transAxes,
                        fontsize=12,
                        color='orange'
                    )
                    continue

                x_grid, y_grid, z_grid = np.meshgrid(x_array, y_array, z_array, indexing='ij')

                plot_diff_mean_histogram(
                    x_grid, y_grid, z_grid,
                    model_name=model_name,
                    bar_label=model_name,
                    mean_type=mean_type,
                    ax=ax
                )
            except Exception as plot_error:
                logger.error(
                    'Failed to plot difference histogram for %s: %s',
                    model_name,
                    plot_error,
                    exc_info=True
                )
                ax.text(
                    0.5, 0.5,
                    f"Error plotting {model_name}",
                    ha='center',
                    va='center',
                    transform=ax.transAxes,
                    fontsize=12,
                    color='red'
                )

        # Hide unused subplots
        for idx in range(num_models, len(axs_flat)):
            unused_ax: Axes = axs_flat[idx]
            unused_ax.axis('off')

        # Adjust layout and title
        fig.suptitle(
            f'{mean_type.capitalize()} Mean Differences vs Baseline - {dataset_name}',
            fontsize=16,
            fontweight='bold',
            y=0.995
        )
        plt.tight_layout(rect=(0, 0, 1, 0.99))

        # Save plots
        plots_dir = os.path.join(PATHS["results_dir"], "plots")
        os.makedirs(plots_dir, exist_ok=True)

        plot_filename = f"differences_mean_histograms_{mean_type}_{dataset_name}.png"
        plot_path = os.path.join(plots_dir, plot_filename)

        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info('Difference histograms plot saved to: %s', plot_path)

        return plot_path

    except Exception as fig_error:
        logger.error(
            'Failed to create %s difference histograms: %s',
            mean_type,
            fig_error,
            exc_info=True
        )
        return None
    finally:
        if fig is not None:
            plt.close(fig)


def calculate_topsis_ranking(val, ideal_values: dict, worst_values: dict, weights=None):
    """
    Calculate TOPSIS ranking for models with error handling

    Parameters
    ----------
    val: Dictionary of all model results
    ideal_values: Dict with 'x', 'y', 'z' lists of ideal values
    worst_values: Dict with 'x', 'y', 'z' lists of worst values
    weights: Weights for [x, y, z] metrics (default: equal weights)

    Returns
    -------
    DataFrame with TOPSIS scores and rankings, or None if calculation fails

    """

    if weights is None:
        weights = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3], dtype=float)

    # Validate weights
    if len(weights) != 3:
        logger.error('Weights must have exactly 3 values')
        return None

    if not np.isclose(weights.sum(), 1.0):
        logger.warning('Weights sum to %.4f; normalizing to 1.0.', weights.sum())
        weights = weights / weights.sum()

    # Validate ideal and worst values
    for key in ['x', 'y', 'z']:
        if key not in ideal_values or key not in worst_values:
            logger.error("Missing key '%s' in ideal or worst values.", key)
            return None

    try:
        # Calculate means
        means = {}
        for model_name, result in val.items():
            try:
                means[model_name] = {
                    'mean_x': float(np.mean(result['x'])),
                    'mean_y': float(np.mean(result['y'])),
                    'mean_z': float(np.mean(result['z'])),
                }
            except Exception as mean_error:
                logger.error('Failed to compute means for %s: %s', model_name, mean_error)
                continue

        if not means:
            logger.error('No valid means calculated')
            return None

        # Create DataFrame
        df = pd.DataFrame.from_dict(means, orient='index')

        # Normalize with division by zero protection
        for col in df.columns:
            vec = df[col].values.astype(float)
            norm = np.sqrt((vec ** 2).sum())
            if norm > 1e-10:  # Avoid division by very small numbers
                df[f'r_{col}'] = vec / norm
            else:
                logger.warning('Column %s has near-zero norm, setting normalized values to 0.', col)
                df[f'r_{col}'] = 0

        # Apply weights
        df['v_mean_x'] = df['r_mean_x'] * weights[0]
        df['v_mean_y'] = df['r_mean_y'] * weights[1]
        df['v_mean_z'] = df['r_mean_z'] * weights[2]

        # Normalize ideal and worst values using norms from data
        norm_x = np.sqrt((df['mean_x'].values ** 2).sum())
        norm_y = np.sqrt((df['mean_y'].values ** 2).sum())
        norm_z = np.sqrt((df['mean_z'].values ** 2).sum())

        x_plus = np.mean(ideal_values['x']) / norm_x if norm_x > 1e-10 else 0
        x_minus = np.mean(worst_values['x']) / norm_x if norm_x > 1e-10 else 0

        y_plus = np.mean(ideal_values['y']) / norm_y if norm_y > 1e-10 else 0
        y_minus = np.mean(worst_values['y']) / norm_y if norm_y > 1e-10 else 0

        z_plus = np.mean(ideal_values['z']) / norm_z if norm_z > 1e-10 else 0
        z_minus = np.mean(worst_values['z']) / norm_z if norm_z > 1e-10 else 0

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

        # TOPSIS score with division by zero protection
        def safe_topsis_score(row):
            denominator = row['S_plus'] + row['S_minus']
            if denominator > 1e-10:
                return row['S_minus'] / denominator
            else:
                return 0.5

        df['C'] = df.apply(safe_topsis_score, axis=1)

        df['Rank'] = df['C'].rank(ascending=False, method='min').astype(int)

        return df.sort_values('C', ascending=False)

    except Exception as topsis_error:
        logger.error('TOPSIS calculation failed: %s', topsis_error)
        return None


def run_safeai_compliance_score(raw_vectors=None):
    """
    Main pipeline for SafeAI Compliance Score visualization and ranking.

    Parameters
    ----------
    raw_vectors : dict, optional
        If provided, bypass file loading and use raw vectors:
        {
          "model_name": (x_list, y_list, z_list),
          ...
        }

    Returns
    -------
    (all_models_results, ranking_df)
        all_models_results : dict or None
        ranking_df     : pd.DataFrame or None
    """
    logger.info("SAFE-AI Compliance Score Pipeline")


    # Raw vectors
    if raw_vectors is not None:
        logger.info('Running in Raw Vector Mode (skipping JSON loads)')

        all_models_results = {}

        for model_name, triple in raw_vectors.items():
            if len(triple) != 3:
                raise ValueError(f"Model '{model_name}' must provide (x,y,z) vectors")

            x, y, z = triple
            if not (len(x) == len(y) == len(z)):
                raise ValueError(f"Model '{model_name}': x,y,z must have same length")

            all_models_results[model_name] = {
                'x': x,
                'y': y,
                'z': z,
                'data': {'x_final': x, 'y_final': y, 'z_final': z}
            }

        logger.info('Loaded %d model(s) from raw vectors.', len(all_models_results))

    # Standard mode
    else:
        logger.info('Running in Normal Mode (loading JSON results)')

        try:
            all_models_results = load_all_model_results(
                MODELS_TO_TRAIN,
                DATASET_CONFIG['dataset_name']
            )
        except Exception as load_error:
            logger.error('Failed to load SafeAI results: %s', load_error)
            return None, None

        if not all_models_results:
            logger.error('No SafeAI results found. Run evaluation first')
            return None, None

        logger.info('Loaded SafeAI results for %d model(s)', len(all_models_results))

    # Plot all curves
    logger.info('Plotting all model curves...')
    try:
        plot_all_model_curves(all_models_results, DATASET_CONFIG['dataset_name'])
        logger.info('Model curves plotted successfully')
    except Exception as curve_error:
        logger.error('Failed to plot all model curves: %s', curve_error)

    # Calculate and plot differences
    logger.info('Calculating performance differences...')
    differences = {}
    try:
        differences = calculate_differences(all_models_results, baseline_model='random')

        if differences:
            logger.info('Differences calculated for %d model(s).', len(differences))
            logger.info('Plotting difference curves...')
            try:
                plot_difference_curves(all_models_results, differences, DATASET_CONFIG['dataset_name'])
                logger.info('Difference curves plotted successfully')
            except Exception as diff_plot_error:
                logger.error('Failed to plot difference curves: %s', diff_plot_error)
        else:
            logger.warning("No differences calculated (baseline model 'random' may be missing)")

    except Exception as diff_error:
        logger.error('Failed to calculate differences: %s', diff_error)

    # Plot histograms for each mean type
    mean_types = ['arithmetic', 'geometric', 'quadratic']
    histogram_success_count = 0

    for mean_type in mean_types:
        logger.info('Processing %s mean histograms...', mean_type)

        # Regular histograms
        try:
            plot_mean_histograms(all_models_results, DATASET_CONFIG['dataset_name'], mean_type)
            histogram_success_count += 1
            logger.info('%s histograms plotted', mean_type.capitalize())
        except Exception as hist_error:
            logger.error('Failed to plot %s histograms: %s', mean_type, hist_error)

        # Difference histograms
        if differences:
            try:
                plot_diff_mean_histograms(differences, DATASET_CONFIG['dataset_name'], mean_type)
                logger.info('%s difference histograms plotted.', mean_type.capitalize())
            except Exception as diff_hist_error:
                logger.error('Failed to plot %s difference histograms: %s', mean_type, diff_hist_error)

    logger.info(
        'Histogram summary: %d/%d mean types completed successfully',
        histogram_success_count,
        len(mean_types),
    )

    # TOPSIS Ranking
    logger.info('TOPSIS Ranking')

    ranking_df = None
    try:
        ranking_df = calculate_topsis_ranking(
            all_models_results,
            ideal_values=TOPSIS_IDEAL_VALUES,
            worst_values=TOPSIS_WORST_VALUES
        )

        if ranking_df is not None and not ranking_df.empty:
            logger.info('TOPSIS Ranking Results:')
            print('\nTOPSIS Ranking Results:')
            print(ranking_df[['mean_x', 'mean_y', 'mean_z', 'C', 'Rank']])

            # Save ranking
            try:
                os.makedirs(PATHS['results_dir'], exist_ok=True)
                ranking_path = os.path.join(
                    PATHS['results_dir'],
                    f'safeai_compliance_ranking_{DATASET_CONFIG['dataset_name']}.csv'
                )
                ranking_df.to_csv(ranking_path)
                logger.info('Ranking saved to: %s', ranking_path)
            except Exception as save_error:
                logger.error('Failed to save ranking CSV: %s', save_error)
        else:
            logger.error('TOPSIS ranking returned None or empty DataFrame.')

    except Exception as topsis_error:
        logger.error('TOPSIS ranking failed: %s', topsis_error)

    # Final summary
    logger.info('SAFE-AI Compliance Pipeline Complete')

    # Summary statistics
    logger.info('Summary:')
    logger.info('  • Models processed: %d', len(all_models_results))
    logger.info('  • Differences calculated: %d', len(differences))
    logger.info(
        '  • Histogram types succeeded: %d/%d',
        histogram_success_count,
        len(mean_types),
    )
    logger.info('  • TOPSIS ranking: %s', 'Success' if ranking_df is not None else 'Failed')
    logger.info('Results directory: %s', PATHS['results_dir'])

    if ranking_df is None:
        logger.warning('Pipeline completed with issues – TOPSIS ranking unavailable.')
    else:
        logger.info('Pipeline completed successfully.')

    return all_models_results, ranking_df


if __name__ == '__main__':
    x_vec = [
        0.6377,
        0.6352,
        0.6279,
        0.6157,
        0.5987,
        0.5768,
        0.5500,
        0.5183,
        0.4820,
        0.4431,
        0.4023,
        0.3596,
        0.3149,
        0.2683,
        0.2198,
        0.1692,
        0.1168,
        0.0662,
        0.0294,
        0.0074,
        0.0000
    ]
    y_vec = [
        1.0000,
        0.6404,
        0.6413,
        0.6425,
        0.6446,
        0.6447,
        0.6447,
        0.6447,
        0.6436,
        0.6416,
        0.6400,
        0.6362,
        0.6292,
        0.6235,
        0.6200,
        0.6105,
        0.6166,
        0.5941,
        0.5000,
        0.5000,
        0.5000
    ]
    z_vec = [
        1.0000,
        0.4953,
        0.5029,
        0.5072,
        0.5020,
        0.5088,
        0.5047,
        0.4942,
        0.5113,
        0.4888,
        0.5069,
        0.5157,
        0.5066,
        0.5013,
        0.4996,
        0.4938,
        0.5169,
        0.5172,
        0.5001,
        0.5024,
        0.5229
    ]
    models_results, ranking = run_safeai_compliance_score(raw_vectors={'llm_rulex': (x_vec, y_vec, z_vec)})