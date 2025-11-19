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

from src.config import DATASET_CONFIG, MODELS_TO_TRAIN, PATHS, MODELS_REQUIRING_SCALING
from safeai_files.check_compliance import safeai_values

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

    model_path = os.path.join(PATHS["models_dir"], f"{model_name}_{dataset_name}.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    logger.info(f"Loading model: {model_path}")

    try:
        model = joblib.load(model_path)
        return model
    except Exception as load_error:
        logger.error(f"Failed to load model {model_name}: {str(load_error)}")
        raise


def load_test_data(dataset_name):
    """
    Load test data splits

    Parameters
    ----------
    dataset_name: Name of the dataset

    Returns
    -------
    Tuple of (x_train, x_test, y_train, y_test)

    """

    logger.info("\nLoading test data...")

    try:
        x_train = pd.read_csv(os.path.join(PATHS["clean_data_dir"], f"x_train_{dataset_name}.csv"))
        x_test = pd.read_csv(os.path.join(PATHS["clean_data_dir"], f"x_test_{dataset_name}.csv"))

        # Safer way to load single column targets
        y_train_df = pd.read_csv(os.path.join(PATHS["clean_data_dir"], f"y_train_{dataset_name}.csv"))
        y_test_df = pd.read_csv(os.path.join(PATHS["clean_data_dir"], f"y_test_{dataset_name}.csv"))

        # Handle both single and multi-column cases
        y_train = y_train_df.iloc[:, 0] if y_train_df.shape[1] > 0 else y_train_df.squeeze()
        y_test = y_test_df.iloc[:, 0] if y_test_df.shape[1] > 0 else y_test_df.squeeze()

        logger.info(f"Train set: x={x_train.shape}, y={y_train.shape}")
        logger.info(f"Test set: x={x_test.shape}, y={y_test.shape}")

        return x_train, x_test, y_train, y_test

    except Exception as load_error:
        logger.error(f"Failed to load test data: {str(load_error)}")
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

    logger.info("\nFinding optimal threshold...")

    f1_scores = []
    for threshold in thresholds:
        y_pred_threshold = np.where(y_prob >= threshold, 1, 0)
        f1 = f1_score(y_test, y_pred_threshold)
        f1_scores.append(f1)

    roc_data = pd.DataFrame({
        'threshold': thresholds,
        'f1_score': f1_scores
    })

    # Best F1
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    logger.info(f"Best F1 Score: {best_f1:.4f} at Threshold: {best_threshold:.4f}")

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

    logger.info("Saving ROC curve...")

    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(PATHS["results_dir"], "plots")
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

        plot_path = os.path.join(plots_dir, f"ROC_curve_{model_name}_{dataset_name}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')

        logger.info(f"ROC curve saved: {plot_path}")

    except Exception as plot_error:
        logger.error(f"Failed to save ROC curve: {str(plot_error)}")
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

    logger.info("\n" + "=" * 60)
    logger.info(f"EVALUATING: {model_name.upper()}")
    logger.info("=" * 60)

    # Validate model has required methods
    if not hasattr(model, 'predict'):
        raise AttributeError(f"Model {model_name} does not have predict method")

    if not hasattr(model, 'predict_proba'):
        raise AttributeError(
            f"Model {model_name} does not have predict_proba method. "
            f"This is required for ROC curve and threshold optimization."
        )

    # Make predictions
    try:
        y_pred = model.predict(x_test)
        y_prob_full = model.predict_proba(x_test)

        # Validate binary classification
        if y_prob_full.shape[1] != 2:
            raise ValueError(
                f"Model returns {y_prob_full.shape[1]} classes, "
                f"but evaluation expects binary classification"
            )

        y_prob = y_prob_full[:, 1]  # Probability of positive class

    except Exception as pred_error:
        logger.error(f"Prediction failed: {str(pred_error)}")
        raise

    # Compute basic metrics and ROC curve
    metrics, fpr, tpr, thresholds = compute_basic_metrics(y_test, y_pred, y_prob)

    # Find optimal threshold
    best_threshold, best_f1, roc_data = find_best_threshold(y_test, y_prob, thresholds)

    # Predictions at best threshold
    y_pred_best = np.where(y_prob >= best_threshold, 1, 0)
    cm_best = confusion_matrix(y_test, y_pred_best)

    logger.info("\nPerformance at Optimal Threshold:")
    logger.info(f"  Threshold: {best_threshold:.4f}")
    logger.info(f"  Accuracy:  {accuracy_score(y_test, y_pred_best):.4f}")
    logger.info(f"  Precision: {precision_score(y_test, y_pred_best):.4f}")
    logger.info(f"  Recall:    {recall_score(y_test, y_pred_best):.4f}")
    logger.info(f"  F1 Score:  {best_f1:.4f}")

    logger.info("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred_best))  # Keep print for formatted output
    logger.info("\nConfusion Matrix:")
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

    # Add SafeAI metrics (optional - don't fail if unavailable)
    logger.info("\n" + "=" * 60)
    logger.info("COMPUTING SAFEAI METRICS")
    logger.info("=" * 60)

    try:
        compliance_results = safeai_values(
            x_train, x_test, y_test, y_prob, model,
            f"{model_name}_{dataset_name}",
            os.path.join(PATHS["results_dir"], "plots")
        )
        result['compliance'] = compliance_results
        logger.info("Compliance metrics computed successfully")
    except ImportError as import_error:
        logger.warning(f"SafeAI module not available: {str(import_error)}")
        result['compliance'] = None
    except Exception as compliance_error:
        logger.warning(f"SafeAI metrics computation failed: {str(compliance_error)}")
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

    os.makedirs(PATHS["results_dir"], exist_ok=True)

    results_path = os.path.join(
        PATHS["results_dir"],
        f"{model_name}_{dataset_name}_evaluation.json"
    )

    try:
        with open(results_path, 'w', encoding='utf-8') as file:
            file.write(json.dumps(result, indent=4))

        logger.info(f"\nEvaluation results saved: {results_path}")

    except Exception as save_error:
        logger.error(f"Failed to save evaluation results: {str(save_error)}")
        raise


def run_evaluation():
    """
    Main evaluation pipeline with comprehensive error handling
    """

    logger.info("\n" + "=" * 60)
    logger.info("MODEL EVALUATION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Dataset: {DATASET_CONFIG['dataset_name']}")
    logger.info(f"Models to evaluate: {MODELS_TO_TRAIN}")
    logger.info("=" * 60)

    # Load test data once
    try:
        x_train, x_test, y_train, y_test = load_test_data(DATASET_CONFIG["dataset_name"])
    except Exception as data_error:
        logger.error(f"Failed to load test data: {str(data_error)}")
        raise

    # Evaluate each model
    all_results = {}
    failed_evaluations = []

    for model_name in MODELS_TO_TRAIN:
        logger.info("\n" + "=" * 60)
        logger.info(f"Processing model: {model_name}")
        logger.info("=" * 60)

        try:
            # Load trained model
            model = load_trained_model(model_name, DATASET_CONFIG["dataset_name"])

            # ADD THIS: Apply scaling if model requires it
            if model_name in MODELS_REQUIRING_SCALING:
                logger.info(f"Loading scaler for {model_name}...")
                scaler_path = os.path.join(PATHS["models_dir"], f"scaler_{DATASET_CONFIG['dataset_name']}.joblib")
                scaler = joblib.load(scaler_path)

                x_train_scaled = pd.DataFrame(
                    scaler.transform(x_train),
                    columns=x_train.columns,
                    index=x_train.index
                )
                x_test_scaled = pd.DataFrame(
                    scaler.transform(x_test),
                    columns=x_test.columns,
                    index=x_test.index
                )
                logger.info(f"Applied scaling to test data for {model_name}")

                # Use scaled data for evaluation
                result, y_pred_best, y_prob = evaluate_model(
                    model, model_name, x_train_scaled, x_test_scaled, y_test,
                    DATASET_CONFIG["dataset_name"]
                )
            else:
                # Use unscaled data for evaluation
                result, y_pred_best, y_prob = evaluate_model(
                    model, model_name, x_train, x_test, y_test,
                    DATASET_CONFIG["dataset_name"]
                )

            # Save results
            save_evaluation_results(result, model_name, DATASET_CONFIG["dataset_name"])

            all_results[model_name] = result
            logger.info(f"✓ Successfully evaluated {model_name}")

        except FileNotFoundError as file_error:
            logger.error(f"✗ {str(file_error)}")
            failed_evaluations.append((model_name, "Model file not found"))
            continue

        except AttributeError as attr_error:
            logger.error(f"✗ Model incompatibility: {str(attr_error)}")
            failed_evaluations.append((model_name, "Missing required methods"))
            continue

        except Exception as eval_error:
            logger.error(f"✗ Unexpected error with {model_name}: {str(eval_error)}")
            failed_evaluations.append((model_name, str(eval_error)))
            continue

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)

    if all_results:
        for model_name, result in all_results.items():
            logger.info(f"\n{model_name}:")
            logger.info(f"  ROC AUC: {result['metrics_at_optimal_threshold']['roc_auc']:.4f}")
            logger.info(f"  F1 Score (optimal threshold): {result['metrics_at_optimal_threshold']['f1_score']:.4f}")
            logger.info(f"  Optimal Threshold: {result['optimal_threshold']:.4f}")
    else:
        logger.warning("No models were successfully evaluated!")

    if failed_evaluations:
        logger.warning("\nFailed evaluations:")
        for failed_model_name, error_msg in failed_evaluations:
            logger.warning(f"  - {failed_model_name}: {error_msg}")

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"\nEvaluated {len(all_results)} model(s) successfully")

    if failed_evaluations:
        logger.info(f"Failed to evaluate {len(failed_evaluations)} model(s)")

    logger.info(f"Results saved in: {PATHS['results_dir']}")

    return all_results, failed_evaluations


if __name__ == "__main__":
    try:
        results, failed = run_evaluation()

        if failed:
            logger.warning(f"\nEvaluation completed with {len(failed)} failure(s)")
        else:
            logger.info("\nAll models evaluated successfully!")

    except Exception as pipeline_error:
        logger.error(f"Evaluation pipeline failed: {str(pipeline_error)}")
        raise