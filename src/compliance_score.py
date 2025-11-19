import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

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
        PATHS["results_dir"],
        f"{model_name}_{dataset_name}_evaluation.json"
    )

    if not os.path.exists(file_path):
        logger.warning(f"Evaluation file not found for {model_name}: {file_path}")
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

            # Extract compliance data
            compliance = data.get('compliance', None)

            if compliance is None:
                logger.warning(f"No compliance data found for {model_name}")
                return None

            return {
                'x_final': compliance.get('x_final'),
                'y_final': compliance.get('y_final'),
                'z_final': compliance.get('z_final'),
            }

    except json.JSONDecodeError as json_error:
        logger.error(f"Invalid JSON in {file_path}: {str(json_error)}")
        return None
    except Exception as load_error:
        logger.error(f"Error loading {file_path}: {str(load_error)}")
        return None


def extract_safeai_values(safeai_data):
    """
    Extract x_final, y_final, z_final from SafeAI results with validation

    Parameters
    ----------
    safeai_data: Dictionary with SafeAI metrics

    Returns
    -------
    Tuple of (x, y, z) or (None, None, None) if not found or invalid

    """

    if safeai_data is None:
        return None, None, None

    x = safeai_data.get('x_final', None)
    y = safeai_data.get('y_final', None)
    z = safeai_data.get('z_final', None)

    # Validate that values are lists/arrays and not empty
    if x is not None and not isinstance(x, (list, np.ndarray)):
        logger.warning(f"x_final is not a list/array: {type(x)}")
        x = None
    if y is not None and not isinstance(y, (list, np.ndarray)):
        logger.warning(f"y_final is not a list/array: {type(y)}")
        y = None
    if z is not None and not isinstance(z, (list, np.ndarray)):
        logger.warning(f"z_final is not a list/array: {type(z)}")
        z = None

    # Check if arrays are empty
    if x is not None and len(x) == 0:
        logger.warning("x_final is empty")
        x = None
    if y is not None and len(y) == 0:
        logger.warning("y_final is empty")
        y = None
    if z is not None and len(z) == 0:
        logger.warning("z_final is empty")
        z = None

    return x, y, z


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
                    logger.info(f"✓ Loaded SafeAI data for {model_name}")
                else:
                    logger.warning(f"{model_name}: Mismatched array lengths (x:{len(x)}, y:{len(y)}, z:{len(z)})")
                    failed_models.append(model_name)
            else:
                logger.warning(f"{model_name}: Incomplete SafeAI data (missing x, y, or z)")
                failed_models.append(model_name)

        except Exception as model_error:
            logger.error(f"Failed to load {model_name}: {str(model_error)}")
            failed_models.append(model_name)
            continue

    if failed_models:
        logger.warning(f"Failed to load {len(failed_models)} model(s): {failed_models}")

    return all_results


def calculate_differences(models_results, baseline_model='random'):
    """
    Calculate performance differences relative to baseline

    Parameters
    ----------
    models_results: Dictionary of all model results
    baseline_model: Name of baseline model (default: 'random')

    Returns
    -------
    Dictionary with difference values, or empty dict if baseline not found

    """

    differences = {}

    if baseline_model not in models_results:
        logger.warning(f"Baseline model '{baseline_model}' not found in results")
        return differences

    baseline_result = models_results[baseline_model]

    if any(v is None for v in [baseline_result['x'], baseline_result['y'], baseline_result['z']]):
        logger.warning(f"Baseline model '{baseline_model}' has invalid data")
        return differences

    try:
        baseline_x = np.array(baseline_result['x'])
        baseline_y = np.array(baseline_result['y'])
        baseline_z = np.array(baseline_result['z'])

        for model_name, result in models_results.items():
            if model_name == baseline_model:
                continue

            try:
                model_x = np.array(result['x'])
                model_y = np.array(result['y'])
                model_z = np.array(result['z'])

                # Validate shapes match
                if (model_x.shape != baseline_x.shape or
                        model_y.shape != baseline_y.shape or
                        model_z.shape != baseline_z.shape):
                    logger.warning(f"Shape mismatch for {model_name}, skipping difference calculation")
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
                logger.error(f"Failed to calculate differences for {model_name}: {str(diff_error)}")
                continue

    except Exception as baseline_error:
        logger.error(f"Failed to process baseline model: {str(baseline_error)}")
        return {}

    return differences


def plot_all_model_curves(models_results, dataset_name):
    """
    Plot all model curves with error handling

    Parameters
    ----------
    models_results: Dictionary of all model results
    dataset_name: Name of the dataset

    """

    if not models_results:
        logger.warning("No model results to plot")
        return

    try:
        num_models = len(models_results)
        cols = 2
        rows = (num_models + 1) // cols

        fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows == 1 and cols == 1:
            axs = np.array([axs])
        axs = axs.flatten()

        # Get x_rga length from first model
        first_result = next(iter(models_results.values()))
        x_rga = np.linspace(0, 1, len(first_result['y']))

        for idx, (model_name, result) in enumerate(models_results.items()):
            try:
                plot_model_curves(
                    x_rga,
                    [result['x'], result['y'], result['z']],
                    model_name=model_name,
                    title=f"{model_name} Curves ({dataset_name})",
                    ax=axs[idx]
                )
            except Exception as plot_error:
                logger.error(f"Failed to plot curves for {model_name}: {str(plot_error)}")
                axs[idx].text(0.5, 0.5, f"Error plotting {model_name}",
                              ha='center', va='center', transform=axs[idx].transAxes)

        # Hide unused subplots
        for idx in range(num_models, len(axs)):
            axs[idx].set_visible(False)

        fig.suptitle(f'All Model Curves - {dataset_name}', fontsize=14, fontweight='bold')
        fig.subplots_adjust(top=0.95, hspace=0.4)
        plt.tight_layout()

        # Save plot
        plots_dir = os.path.join(PATHS["results_dir"], "plots")
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir, f"all_model_curves_{dataset_name}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved: {plot_path}")

    except Exception as fig_error:
        logger.error(f"Failed to create model curves plot: {str(fig_error)}")
    finally:
        plt.close()


def plot_difference_curves(models_results, differences, dataset_name):
    """
    Plot performance differences relative to baseline with error handling

    Parameters
    ----------
    models_results: Dictionary of all model results
    differences: Dictionary of difference values
    dataset_name: Name of the dataset

    """

    if not differences:
        logger.warning("No difference data to plot")
        return

    try:
        num_models = len(differences)
        cols = 2
        rows = (num_models + 1) // cols

        fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows == 1 and cols == 1:
            axs = np.array([axs])
        axs = axs.flatten()

        # Get x_rga length from first model
        first_result = next(iter(models_results.values()))
        x_rga = np.linspace(0, 1, len(first_result['y']))

        for idx, (model_name, diff) in enumerate(differences.items()):
            try:
                plot_model_curves(
                    x_rga,
                    [diff['x'], diff['y'], diff['z']],
                    model_name="Baseline",
                    prefix="Difference",
                    title=f"{model_name} Difference vs Baseline ({dataset_name})",
                    ax=axs[idx]
                )
            except Exception as plot_error:
                logger.error(f"Failed to plot difference for {model_name}: {str(plot_error)}")
                axs[idx].text(0.5, 0.5, f"Error plotting {model_name}",
                              ha='center', va='center', transform=axs[idx].transAxes)

        # Hide unused subplots
        for idx in range(num_models, len(axs)):
            axs[idx].set_visible(False)

        fig.suptitle(f'Performance Differences vs Baseline - {dataset_name}',
                     fontsize=14, fontweight='bold')
        fig.subplots_adjust(top=0.95, hspace=0.4)
        plt.tight_layout()

        # Save plot
        plots_dir = os.path.join(PATHS["results_dir"], "plots")
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir, f"difference_curves_{dataset_name}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved: {plot_path}")

    except Exception as fig_error:
        logger.error(f"Failed to create difference curves plot: {str(fig_error)}")
    finally:
        plt.close()


def plot_mean_histograms(models_results, dataset_name, mean_type="arithmetic"):
    """
    Plot mean histograms for all models with error handling

    Parameters
    ----------
    models_results: Dictionary of all model results
    dataset_name: Name of the dataset
    mean_type: "arithmetic", "geometric", or "quadratic"

    """

    if not models_results:
        logger.warning("No model results for histograms")
        return

    try:
        num_models = len(models_results)
        cols = 2
        rows = (num_models + 1) // cols

        fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows == 1 and cols == 1:
            axs = np.array([axs])
        axs = axs.flatten()

        for idx, (model_name, result) in enumerate(models_results.items()):
            try:
                # Create meshgrid
                x_array = np.array(result['x'])
                y_array = np.array(result['y'])
                z_array = np.array(result['z'])

                # Validate arrays are not empty
                if x_array.size == 0 or y_array.size == 0 or z_array.size == 0:
                    logger.warning(f"Empty arrays for {model_name}, skipping histogram")
                    axs[idx].text(0.5, 0.5, f"No data for {model_name}",
                                  ha='center', va='center', transform=axs[idx].transAxes)
                    continue

                x_grid, y_grid, z_grid = np.meshgrid(x_array, y_array, z_array, indexing='ij')

                plot_mean_histogram(
                    x_grid, y_grid, z_grid,
                    model_name=model_name,
                    bar_label=model_name,
                    mean_type=mean_type,
                    ax=axs[idx]
                )
            except Exception as plot_error:
                logger.error(f"Failed to plot histogram for {model_name}: {str(plot_error)}")
                axs[idx].text(0.5, 0.5, f"Error plotting {model_name}",
                              ha='center', va='center', transform=axs[idx].transAxes)

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
        logger.info(f"Plot saved: {plot_path}")

    except Exception as fig_error:
        logger.error(f"Failed to create {mean_type} histograms: {str(fig_error)}")
    finally:
        plt.close()


def plot_diff_mean_histograms(differences, dataset_name, mean_type="arithmetic"):
    """
    Plot mean histograms for performance differences vs baseline

    Parameters
    ----------
    differences: Dictionary of difference values
    dataset_name: Name of the dataset
    mean_type: "arithmetic", "geometric", or "quadratic"

    """

    if not differences:
        logger.info(f"No difference data available, skipping {mean_type} difference histograms")
        return

    try:
        num_models = len(differences)
        cols = 2
        rows = (num_models + 1) // cols

        fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows == 1 and cols == 1:
            axs = np.array([axs])
        axs = axs.flatten()

        for idx, (model_name, diff) in enumerate(differences.items()):
            try:
                # Create meshgrid
                x_array = np.array(diff['x'])
                y_array = np.array(diff['y'])
                z_array = np.array(diff['z'])

                if x_array.size == 0 or y_array.size == 0 or z_array.size == 0:
                    logger.warning(f"Empty difference arrays for {model_name}")
                    axs[idx].text(0.5, 0.5, f"No data for {model_name}",
                                  ha='center', va='center', transform=axs[idx].transAxes)
                    continue

                x_grid, y_grid, z_grid = np.meshgrid(x_array, y_array, z_array, indexing='ij')

                plot_diff_mean_histogram(
                    x_grid, y_grid, z_grid,
                    model_name=model_name,
                    bar_label=model_name,
                    mean_type=mean_type,
                    ax=axs[idx]
                )
            except Exception as plot_error:
                logger.error(f"Failed to plot difference histogram for {model_name}: {str(plot_error)}")
                axs[idx].text(0.5, 0.5, f"Error plotting {model_name}",
                              ha='center', va='center', transform=axs[idx].transAxes)

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
        logger.info(f"Plot saved: {plot_path}")

    except Exception as fig_error:
        logger.error(f"Failed to create {mean_type} difference histograms: {str(fig_error)}")
    finally:
        plt.close()


def calculate_topsis_ranking(models_results, ideal_values: dict, worst_values: dict, weights=None):
    """
    Calculate TOPSIS ranking for models with error handling

    Parameters
    ----------
    models_results: Dictionary of all model results
    ideal_values: Dict with 'x', 'y', 'z' lists of ideal values
    worst_values: Dict with 'x', 'y', 'z' lists of worst values
    weights: Weights for [x, y, z] metrics (default: equal weights)

    Returns
    -------
    DataFrame with TOPSIS scores and rankings, or None if calculation fails

    """

    if weights is None:
        weights = np.array([1 / 3, 1 / 3, 1 / 3])

    # Validate weights
    if len(weights) != 3:
        logger.error("Weights must have exactly 3 values")
        return None

    if not np.isclose(weights.sum(), 1.0):
        logger.warning(f"Weights sum to {weights.sum()}, normalizing to 1.0")
        weights = weights / weights.sum()

    # Validate ideal and worst values
    for key in ['x', 'y', 'z']:
        if key not in ideal_values or key not in worst_values:
            logger.error(f"Missing '{key}' in ideal or worst values")
            return None

    try:
        # Calculate means
        means = {}
        for model_name, result in models_results.items():
            try:
                means[model_name] = {
                    'mean_x': float(np.mean(result['x'])),
                    'mean_y': float(np.mean(result['y'])),
                    'mean_z': float(np.mean(result['z'])),
                }
            except Exception as mean_error:
                logger.error(f"Failed to calculate means for {model_name}: {str(mean_error)}")
                continue

        if not means:
            logger.error("No valid means calculated")
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
                logger.warning(f"Column {col} has zero or near-zero norm, setting normalized values to 0")
                df[f'r_{col}'] = 0

        # Apply weights
        df['v_mean_x'] = df['r_mean_x'] * weights[0]
        df['v_mean_y'] = df['r_mean_y'] * weights[1]
        df['v_mean_z'] = df['r_mean_z'] * weights[2]

        # Normalize ideal and worst values using norms from data
        norm_x = np.sqrt((df["mean_x"].values ** 2).sum())
        norm_y = np.sqrt((df["mean_y"].values ** 2).sum())
        norm_z = np.sqrt((df["mean_z"].values ** 2).sum())

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
        logger.error(f"TOPSIS calculation failed: {str(topsis_error)}")
        return None


def run_safeai_compliance_score():
    """
    Main pipeline for SafeAI Compliance Score with comprehensive error handling

    Returns
    -------
    Tuple of (models_results dict, ranking DataFrame) or (None, None) if fails
    """
    logger.info("\n" + "=" * 60)
    logger.info("SafeAI METRICS VISUALIZATION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Dataset: {DATASET_CONFIG['dataset_name']}")
    logger.info(f"Models: {MODELS_TO_TRAIN}")
    logger.info("=" * 60)

    # Validate config
    try:
        if not isinstance(TOPSIS_IDEAL_VALUES, dict) or not isinstance(TOPSIS_WORST_VALUES, dict):
            logger.error("TOPSIS_IDEAL_VALUES and TOPSIS_WORST_VALUES must be dictionaries")
            return None, None
        for key in ['x', 'y', 'z']:
            if key not in TOPSIS_IDEAL_VALUES:
                logger.error(f"TOPSIS_IDEAL_VALUES missing key: {key}")
                return None, None
            if key not in TOPSIS_WORST_VALUES:
                logger.error(f"TOPSIS_WORST_VALUES missing key: {key}")
                return None, None
    except NameError as config_error:
        logger.error(f"TOPSIS configuration not found in config: {str(config_error)}")
        return None, None

    # Load all model results
    logger.info("\nLoading SafeAI results...")
    try:
        models_results = load_all_model_results(MODELS_TO_TRAIN, DATASET_CONFIG['dataset_name'])
    except Exception as load_error:
        logger.error(f"Failed to load model results: {str(load_error)}")
        return None, None

    if not models_results:
        logger.error("No SafeAI results found! Ensure models have been evaluated first.")
        logger.error("Run evaluation.py before running SafeAI compliance scoring.")
        return None, None

    logger.info(f"✓ Loaded {len(models_results)} model(s) with valid SafeAI data")

    # Plot all curves
    logger.info("\nPlotting all model curves...")
    try:
        plot_all_model_curves(models_results, DATASET_CONFIG['dataset_name'])
        logger.info("✓ Model curves plotted successfully")
    except Exception as curve_error:
        logger.error(f"✗ Failed to plot model curves: {str(curve_error)}")

    # Calculate and plot differences
    logger.info("\nCalculating performance differences...")
    differences = {}
    try:
        differences = calculate_differences(models_results, baseline_model='random')

        if differences:
            logger.info(f"✓ Calculated differences for {len(differences)} model(s)")
            logger.info("Plotting difference curves...")
            try:
                plot_difference_curves(models_results, differences, DATASET_CONFIG['dataset_name'])
                logger.info("✓ Difference curves plotted successfully")
            except Exception as diff_plot_error:
                logger.error(f"✗ Failed to plot difference curves: {str(diff_plot_error)}")
        else:
            logger.warning("⚠ No differences calculated (baseline model 'random' may be missing)")

    except Exception as diff_error:
        logger.error(f"✗ Failed to calculate differences: {str(diff_error)}")

    # Plot histograms for each mean type
    mean_types = ['arithmetic', 'geometric', 'quadratic']
    histogram_success_count = 0

    for mean_type in mean_types:
        logger.info(f"\nProcessing {mean_type} mean histograms...")

        # Regular histograms
        try:
            plot_mean_histograms(models_results, DATASET_CONFIG['dataset_name'], mean_type)
            histogram_success_count += 1
            logger.info(f"✓ {mean_type.capitalize()} histograms plotted")
        except Exception as hist_error:
            logger.error(f"✗ Failed to plot {mean_type} histograms: {str(hist_error)}")

        # Difference histograms
        if differences:
            try:
                plot_diff_mean_histograms(differences, DATASET_CONFIG['dataset_name'], mean_type)
                logger.info(f"✓ {mean_type.capitalize()} difference histograms plotted")
            except Exception as diff_hist_error:
                logger.error(f"✗ Failed to plot {mean_type} difference histograms: {str(diff_hist_error)}")

    logger.info(f"\nHistogram summary: {histogram_success_count}/{len(mean_types)} types completed successfully")

    # TOPSIS Ranking
    logger.info("\n" + "=" * 60)
    logger.info("TOPSIS RANKING")
    logger.info("=" * 60)

    ranking_df = None
    try:
        ranking_df = calculate_topsis_ranking(
            models_results,
            ideal_values=TOPSIS_IDEAL_VALUES,
            worst_values=TOPSIS_WORST_VALUES
        )

        if ranking_df is not None and not ranking_df.empty:
            logger.info("\n✓ TOPSIS Ranking Results:")
            print("\n", ranking_df[['mean_x', 'mean_y', 'mean_z', 'C', 'Rank']])

            # Save ranking
            try:
                os.makedirs(PATHS["results_dir"], exist_ok=True)
                ranking_path = os.path.join(
                    PATHS["results_dir"],
                    f"safeai_compliance_ranking_{DATASET_CONFIG['dataset_name']}.csv"
                )
                ranking_df.to_csv(ranking_path)
                logger.info(f"\n✓ Ranking saved: {ranking_path}")
            except Exception as save_error:
                logger.error(f"✗ Failed to save ranking: {str(save_error)}")
        else:
            logger.error("✗ TOPSIS ranking calculation returned None or empty DataFrame")

    except Exception as topsis_error:
        logger.error(f"✗ TOPSIS ranking failed: {str(topsis_error)}")
        import traceback
        logger.debug(traceback.format_exc())

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)

    # Summary statistics
    logger.info(f"\nSummary:")
    logger.info(f"  • Models processed: {len(models_results)}")
    logger.info(f"  • Differences calculated: {len(differences)}")
    logger.info(f"  • Histogram types completed: {histogram_success_count}/{len(mean_types)}")
    logger.info(f"  • TOPSIS ranking: {'✓ Success' if ranking_df is not None else '✗ Failed'}")
    logger.info(f"\n  Results directory: {PATHS['results_dir']}")

    # Check if we have minimum viable results
    if ranking_df is None:
        logger.warning("\n⚠ Pipeline completed with issues - TOPSIS ranking unavailable")
    else:
        logger.info("\n✓ Pipeline completed successfully!")

    return models_results, ranking_df


if __name__ == "__main__":
    try:
        models_results, ranking = run_safeai_compliance_score()

        if models_results is None:
            logger.error("\n✗ Pipeline failed - no model results loaded")
            exit(1)
        elif ranking is None:
            logger.warning("\n⚠ Pipeline completed partially - rankings unavailable")
            exit(0)
        else:
            logger.info("\n✓ All operations completed successfully!")
            exit(0)

    except KeyboardInterrupt:
        logger.warning("\n⚠ Pipeline interrupted by user")
        exit(130)
    except Exception as pipeline_error:
        logger.error(f"\n✗ SafeAI compliance pipeline failed: {str(pipeline_error)}")
        import traceback

        logger.debug(traceback.format_exc())
        raise