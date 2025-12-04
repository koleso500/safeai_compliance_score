import joblib
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import logging
from sklearn.metrics import (
    accuracy_score, auc, classification_report, confusion_matrix,
    f1_score, roc_curve, precision_score, recall_score
)
from sklearn.model_selection import train_test_split

from src.config import DATASET_CONFIG, MODELS_TO_TRAIN, PATHS, MODELS_REQUIRING_SCALING
from safeai_files.check_compliance import safeai_values
from src.models import get_required_base_models
from src.train_models import load_base_models_for_ensemble

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_trained_model(model_name, dataset_name):
    """
    Load a trained model from disk

    Parameters
    ----------
    model_name: Name of the model
    dataset_name: Name of the dataset

    Returns
    -------
    Trained model

    """
    model_path = os.path.join(PATHS['models_dir'], f'{model_name}_{dataset_name}.joblib')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model not found: {model_path}')

    logger.info('Loading model from: %s', model_path)

    try:
        model = joblib.load(model_path)
        return model
    except Exception as load_error:
        logger.error('Failed to load model %s: %s', model_name, load_error)
        raise


def load_test_data(dataset_name):
    """
    Load train/test data splits

    Parameters
    ----------
    dataset_name: Name of the dataset

    Returns
    -------
    Tuple of (x_train, x_test, y_train, y_test)

    """
    logger.info('Loading train/test splits...')

    try:
        x_train = pd.read_csv(os.path.join(PATHS['clean_data_dir'], f'x_train_{dataset_name}.csv'))
        x_test = pd.read_csv(os.path.join(PATHS['clean_data_dir'], f'x_test_{dataset_name}.csv'))

        # Safer way to load single column targets
        y_train_df = pd.read_csv(os.path.join(PATHS['clean_data_dir'], f'y_train_{dataset_name}.csv'))
        y_test_df = pd.read_csv(os.path.join(PATHS['clean_data_dir'], f'y_test_{dataset_name}.csv'))

        y_train = y_train_df.iloc[:, 0]
        y_test = y_test_df.iloc[:, 0]

        logger.info('Train set: %s', x_train.shape)
        logger.info('Test  set: %s', x_test.shape)

        return x_train, x_test, y_train, y_test

    except Exception as load_error:
        logger.error('Failed to load test data: %s', load_error)
        raise


def compute_basic_metrics(y_test, y_pred, y_prob):

    """

    Parameters
    ----------
    y_test: True labels
    y_pred: Predicted labels
    y_prob: Predicted probabilities

    Returns
    -------
    Dictionary of metrics

    """
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
    }

    # ROC AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    metrics['roc_auc'] = auc(fpr, tpr)
    metrics['fpr'] = fpr.tolist()
    metrics['tpr'] = tpr.tolist()
    metrics['thresholds'] = thresholds.tolist()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = cm.tolist()

    return metrics, fpr, tpr, thresholds


def find_best_threshold(y_test, y_prob, thresholds):
    """
    Find the best threshold based on F1 score

    Parameters
    ----------
    y_test: True labels
    y_prob: Predicted probabilities
    thresholds: Thresholds from ROC curve

    Returns
    -------
    Tuple of (best_threshold, best_f1, roc_data DataFrame)

    """
    logger.info('Searching for optimal threshold based on F1 score...')

    f1_scores = []
    for threshold in thresholds:
        y_pred_threshold = np.where(y_prob >= threshold, 1, 0)
        f1 = f1_score(y_test, y_pred_threshold)
        f1_scores.append(f1)

    roc_data = pd.DataFrame({'threshold': thresholds, 'f1_score': f1_scores})

    # Best F1
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    logger.info('Best F1 Score: %.4f at threshold: %.4f', best_f1, best_threshold)

    return best_threshold, best_f1, roc_data


def plot_roc_curve(fpr, tpr, roc_auc, model_name, dataset_name):
    """
    Plot and save ROC curve

    Parameters
    ----------
    fpr: False positive rates
    tpr: True positive rates
    roc_auc: AUC score
    model_name: Name of the model
    dataset_name: Name of the dataset

    """
    logger.info('Saving ROC curve...')

    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(PATHS['results_dir'], 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    try:
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name} ({dataset_name})', fontsize=14)
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)

        plot_path = os.path.join(plots_dir, f'ROC_curve_{model_name}_{dataset_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')

        logger.info('ROC curve saved to: %s', plot_path)

    except Exception as plot_error:
        logger.error('Failed to save ROC curve for %s: %s', model_name, plot_error)
        raise
    finally:
        plt.close()  # Always close the figure


def evaluate_model(model, model_name, x_train, x_test, y_test, dataset_name):
    """
    Evaluate a trained model on test data

    Parameters
    ----------
    model: Trained model
    model_name: Name of the model
    x_train: Training features (used for SafeAI compliance)
    x_test: Test features
    y_test: Test target
    dataset_name: Name of the dataset

    Returns
    -------
    Tuple of (results dict, predictions, probabilities)

    """
    logger.info('EVALUATING MODEL: %s', model_name.upper())

    # Validate model has required methods
    if not hasattr(model, 'predict'):
        raise AttributeError(f'Model {model_name} does not have predict() method')

    if not hasattr(model, 'predict_proba'):
        raise AttributeError(
            f'Model {model_name} does not have predict_proba method.'
            f'This is required for ROC curve and threshold optimization.'
        )

    # Make predictions
    try:
        if model_name.lower() == 'sem':
            logger.info('Generating meta-features for %s...', model_name.upper())

            # Load base models used in training
            required_bases = get_required_base_models(model_name)
            base_models = load_base_models_for_ensemble(required_bases, dataset_name)

            # Build meta-features for TEST set
            meta_test = {}
            for base_name, base_model in base_models.items():
                if hasattr(base_model, "predict_proba"):
                    meta_test[f"{base_name}_pred"] = base_model.predict_proba(x_test)[:, 1]
                else:
                    meta_test[f"{base_name}_pred"] = base_model.predict(x_test)

            meta_test = pd.DataFrame(meta_test)

            # Predict using only the final estimator
            y_prob_full = model.final_estimator_.predict_proba(meta_test)
            y_prob = y_prob_full[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

        # For other models
        else:
            y_pred = model.predict(x_test)
            y_prob_full = model.predict_proba(x_test)

            if y_prob_full.shape[1] != 2:
                raise ValueError(
                    f'Model {model_name} returned probabilities for '
                    f'{y_prob_full.shape[1]} classes but binary classification expected.'
                )
            y_prob = y_prob_full[:, 1]

    except Exception as pred_error:
        logger.error('Prediction failed for %s: %s', model_name, pred_error)
        raise

    # Compute basic metrics and ROC curve
    metrics, fpr, tpr, thresholds = compute_basic_metrics(y_test, y_pred, y_prob)

    # Find optimal threshold
    best_threshold, best_f1, roc_data = find_best_threshold(y_test, y_prob, thresholds)

    # Predictions at best threshold
    y_pred_best = np.where(y_prob >= best_threshold, 1, 0)
    cm_best = confusion_matrix(y_test, y_pred_best)

    logger.info('Performance at optimal threshold (%.4f):', best_threshold)
    logger.info('Accuracy : %.4f', accuracy_score(y_test, y_pred_best))
    logger.info('Precision: %.4f', precision_score(y_test, y_pred_best))
    logger.info('Recall   : %.4f', recall_score(y_test, y_pred_best))
    logger.info('F1 Score : %.4f', best_f1)

    logger.info('Detailed Classification Report (at optimal threshold):')
    print(classification_report(y_test, y_pred_best))  # Keep print for formatted output
    logger.info('\nConfusion Matrix:')
    print(cm_best)  # Keep print for formatted output

    # ROC curve
    plot_roc_curve(fpr, tpr, metrics['roc_auc'], model_name, dataset_name)

    # Results dictionary
    result = {
        'model_name': model_name,
        'dataset_name': dataset_name,
        'optimal_threshold': float(best_threshold),
        'metrics_at_optimal_threshold': {
            'accuracy': float(accuracy_score(y_test, y_pred_best)),
            'precision': float(precision_score(y_test, y_pred_best)),
            'recall': float(recall_score(y_test, y_pred_best)),
            'roc_auc': float(metrics['roc_auc']),
            'f1_score': float(best_f1),
            'confusion_matrix': cm_best.tolist(),
        }
    }

    # Add SafeAI metrics
    logger.info('Computing SAFE-AI Metrics...')

    try:
        compliance_results = safeai_values(
            x_train, x_test, y_test, y_prob, model,
            f'{model_name}_{dataset_name}',
            os.path.join(PATHS['results_dir'], 'plots')
        )
        result['compliance'] = compliance_results
        logger.info('Compliance metrics computed successfully for %s', model_name)
    except ImportError as import_error:
        logger.warning('SafeAI module not available: %s', import_error)
        result['compliance'] = None
    except Exception as compliance_error:
        logger.warning('SafeAI metrics failed for %s: %s', model_name, compliance_error)
        result['compliance'] = None

    return result, y_pred_best, y_prob


def save_evaluation_results(result, model_name, dataset_name):
    """
    Save evaluation results to JSON

    Parameters
    ----------
    result: Dictionary of evaluation results
    model_name: Name of the model
    dataset_name: Name of the dataset

    """
    os.makedirs(PATHS['results_dir'], exist_ok=True)

    results_path = os.path.join(
        PATHS['results_dir'],
        f'{model_name}_{dataset_name}_evaluation.json'
    )

    try:
        with open(results_path, 'w', encoding='utf-8') as file:
            file.write(json.dumps(result, indent=4))

        logger.info('Evaluation results saved to: %s', results_path)

    except Exception as save_error:
        logger.error('Failed to save evaluation results for %s: %s', model_name, save_error)
        raise


QSVC_MAX_EVAL_SAMPLES = 100

def run_evaluation():
    """
    Main evaluation pipeline with comprehensive error handling

    """
    os.makedirs(PATHS['results_dir'], exist_ok=True)
    os.makedirs(PATHS['models_dir'], exist_ok=True)

    logger.info("Model Evaluation Pipeline")
    logger.info('Dataset: %s', DATASET_CONFIG['dataset_name'])
    logger.info('Models to evaluate: %s', MODELS_TO_TRAIN)

    # Load test data once
    try:
        x_train, x_test, y_train, y_test = load_test_data(DATASET_CONFIG['dataset_name'])
    except Exception as data_error:
        logger.error('Failed to load test data: %s', data_error)
        raise

    # Evaluate each model
    all_results = {}
    failed_evaluations = []

    for model_name in MODELS_TO_TRAIN:
        logger.info('Processing model: %s', model_name)

        try:
            # Load trained model
            model = load_trained_model(model_name, DATASET_CONFIG['dataset_name'])

            # Downsample train & test only for qsvc
            if model_name == "qsvc":
                logger.info('QSVC detected — applying downsampling for evaluation')

                # Reduce test
                if len(x_test) > QSVC_MAX_EVAL_SAMPLES:
                    x_test_eval, _, y_test_eval, _ = train_test_split(
                        x_test, y_test,
                        train_size=QSVC_MAX_EVAL_SAMPLES,
                        stratify=y_test,
                        random_state=42
                    )
                    logger.info(
                        'Reduced QSVC test set from %d → %d',
                        len(x_test), len(x_test_eval)
                    )
                else:
                    x_test_eval, y_test_eval = x_test, y_test

                # Reduce train (for SAFE-AI robustness)
                if len(x_train) > QSVC_MAX_EVAL_SAMPLES:
                    x_train_eval, _, y_train_eval, _ = train_test_split(
                        x_train, y_train,
                        train_size=QSVC_MAX_EVAL_SAMPLES,
                        stratify=y_train,
                        random_state=42
                    )
                    logger.info(
                        'Reduced QSVC train set from %d → %d',
                        len(x_train), len(x_train_eval)
                    )
                else:
                    x_train_eval, y_train_eval = x_train, y_train

            else:
                # All other models: full train/test
                x_train_eval, y_train_eval = x_train, y_train
                x_test_eval, y_test_eval = x_test, y_test

            # Apply scaling if model requires it
            if model_name in MODELS_REQUIRING_SCALING:
                logger.info('Model %s requires scaling. Loading scaler...', model_name)
                scaler_path = os.path.join(PATHS['models_dir'], f'scaler_{DATASET_CONFIG['dataset_name']}.joblib')
                scaler = joblib.load(scaler_path)

                x_train_scaled = pd.DataFrame(
                    scaler.transform(x_train_eval),
                    columns=x_train.columns,
                    index=x_train_eval.index
                )
                x_test_scaled = pd.DataFrame(
                    scaler.transform(x_test_eval),
                    columns=x_test.columns,
                    index=x_test_eval.index
                )

                logger.info(
                    'Scaling applied for %s (train=%d, test=%d)',
                    model_name, len(x_train_scaled), len(x_test_scaled)
                )

                # Use scaled data for evaluation
                result, y_pred_best, y_prob = evaluate_model(
                    model, model_name,
                    x_train_scaled, x_test_scaled,
                    y_test_eval,
                    DATASET_CONFIG['dataset_name']
                )
            else:
                # Use unscaled data for evaluation
                result, y_pred_best, y_prob = evaluate_model(
                    model, model_name,
                    x_train_eval, x_test_eval,
                    y_test_eval,
                    DATASET_CONFIG['dataset_name']
                )

            # Save results
            save_evaluation_results(result, model_name, DATASET_CONFIG['dataset_name'])

            all_results[model_name] = result
            logger.info('Successfully evaluated model: %s', model_name)

        except FileNotFoundError as file_error:
            logger.error('%s', file_error)
            failed_evaluations.append((model_name, 'Model file not found'))
            continue

        except AttributeError as attr_error:
            logger.error('Model incompatibility for %s: %s', model_name, attr_error)
            failed_evaluations.append((model_name, 'Missing required methods'))
            continue

        except Exception as eval_error:
            logger.error('Unexpected error for %s: %s', model_name, eval_error)
            failed_evaluations.append((model_name, str(eval_error)))
            continue

    # Summary
    logger.info('Evaluation Summary')

    if all_results:
        for model_name, result in all_results.items():
            metrics_opt = result['metrics_at_optimal_threshold']
            logger.info(
                '%s: AUC=%.4f, F1(opt)=%.4f, Thr=%.4f',
                model_name,
                metrics_opt['roc_auc'],
                metrics_opt['f1_score'],
                result['optimal_threshold'],
            )
    else:
        logger.warning('No models were successfully evaluated.')

    if failed_evaluations:
        logger.warning('Failed evaluations:')
        for failed_model_name, error in failed_evaluations:
            logger.warning('  - %s: %s', failed_model_name, error)

    logger.info('Evaluation results directory: %s', PATHS['results_dir'])
    logger.info('Evaluation Complete')

    return all_results, failed_evaluations


if __name__ == '__main__':
    try:
        results, failed = run_evaluation()

        if failed:
            logger.warning('Evaluation completed with %d failure(s)', len(failed))
        else:
            logger.info('All models evaluated successfully')

    except Exception as pipeline_error:
        logger.error('Evaluation pipeline failed: %s', pipeline_error)
        raise