import joblib
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score, auc, classification_report, confusion_matrix,
    f1_score, roc_curve, precision_score, recall_score
)

from src.config import DATASET_CONFIG, MODELS_TO_TRAIN, PATHS
from safeai_files.check_compliance import safeai_values



def load_trained_model(model_name, dataset_name):

    """

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

    print(f"Loading model: {model_path}")
    model = joblib.load(model_path)
    return model


def load_test_data(dataset_name):

    """

    Parameters
    ----------
    dataset_name: Name of the dataset

    Returns
    -------
    Tuple of (x_train, x_test, y_train, y_test)

    """

    print("\nLoading test data...")

    x_train = pd.read_csv(os.path.join(PATHS["clean_data_dir"], f"x_train_{dataset_name}.csv"))
    x_test = pd.read_csv(os.path.join(PATHS["clean_data_dir"], f"x_test_{dataset_name}.csv"))
    y_train = pd.read_csv(os.path.join(PATHS["clean_data_dir"], f"y_train_{dataset_name}.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(PATHS["clean_data_dir"], f"y_test_{dataset_name}.csv")).squeeze()

    print(f"Test set: x={x_test.shape}, y={y_test.shape}")
    return x_train, x_test, y_train, y_test


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

    print("\nFinding optimal threshold...")

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

    print(f"Best F1 Score: {best_f1:.4f} at Threshold: {best_threshold:.4f}")

    return best_threshold, best_f1, roc_data


def plot_roc_curve(fpr, tpr, roc_auc, model_name, dataset_name):

    """

    Parameters
    ----------
    fpr: False positive rates
    tpr: True positive rates
    roc_auc: AUC score
    model_name: Name of the model
    dataset_name: Name of the dataset

    Returns
    -------

    """

    print("Saving ROC curve...")

    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(PATHS["results_dir"], "plots")
    os.makedirs(plots_dir, exist_ok=True)

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
    plt.close()

    print(f"ROC curve saved: {plot_path}")


def evaluate_model(model, model_name, x_train, x_test, y_test, dataset_name):

    """

    Parameters
    ----------
    model: Trained model
    model_name: Name of the model
    x_train: Training features
    x_test: Test features
    y_test: Test target
    dataset_name: Name of the dataset

    Returns
    -------
    Dictionary with all evaluation results

    """

    print("\n" + "=" * 60)
    print(f"EVALUATING: {model_name.upper()}")
    print("=" * 60)

    # Make predictions
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    # Compute basic metrics and ROC curve
    metrics, fpr, tpr, thresholds = compute_basic_metrics(y_test, y_pred, y_prob)

    # Find optimal threshold
    best_threshold, best_f1, roc_data = find_best_threshold(y_test, y_prob, thresholds)

    # Predictions at best threshold
    y_pred_best = np.where(y_prob >= best_threshold, 1, 0)
    cm_best = confusion_matrix(y_test, y_pred_best)

    print("\nPerformance at Optimal Threshold:")
    print(f"  Threshold: {best_threshold:.4f}")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred_best):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred_best):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred_best):.4f}")
    print(f"  F1 Score:  {best_f1:.4f}")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred_best))
    print("\nConfusion Matrix:")
    print(cm_best)

    # ROC curve
    plot_roc_curve(fpr, tpr, metrics['roc_auc'], model_name, dataset_name)

    # Results dictionary
    result = {
        'model_name': model_name,
        'dataset_name': dataset_name,
        'optimal_threshold': best_threshold,
        'metrics_at_optimal_threshold': {
            'accuracy': float(accuracy_score(y_test, y_pred_best)),
            'precision': float(precision_score(y_test, y_pred_best)),
            'recall': float(recall_score(y_test, y_pred_best)),
            'roc_auc': metrics['roc_auc'],
            'f1_score': best_f1,
            'confusion_matrix': cm_best.tolist(),
        }
    }

    # Add SafeAI metrics
    print("\n" + "=" * 60)
    print("COMPUTING SAFEAI METRICS")
    print("=" * 60)


    # Compliance metrics
    compliance_results = safeai_values(
        x_train, x_test, y_test, y_prob, model,
        f"{model_name}_{dataset_name}",
        os.path.join(PATHS["results_dir"], "plots")
    )
    result['compliance'] = compliance_results
    print("Compliance metrics computed successfully")

    return result, y_pred_best, y_prob


def save_evaluation_results(result, model_name, dataset_name):

    """
    Save evaluation results to JSON

    Parameters
    ----------
    result: Dictionary of evaluation results
    model_name: Name of the model
    dataset_name: Name of the dataset

    Returns
    -------

    """

    os.makedirs(PATHS["results_dir"], exist_ok=True)

    results_path = os.path.join(
        PATHS["results_dir"],
        f"{model_name}_{dataset_name}_evaluation.json"
    )
    json_str = json.dumps(result, indent=4)
    with open(results_path, 'w', encoding='utf-8') as file:
        file.write(json_str)

    print(f"\nEvaluation results saved: {results_path}")


def run_evaluation():
    """
    Main evaluation pipeline
    """
    print("\n" + "=" * 60)
    print("MODEL EVALUATION PIPELINE")
    print("=" * 60)
    print(f"Dataset: {DATASET_CONFIG['dataset_name']}")
    print(f"Models to evaluate: {MODELS_TO_TRAIN}")
    print("=" * 60)

    # Load test data once
    x_train, x_test, y_train, y_test = load_test_data(DATASET_CONFIG["dataset_name"])

    # Evaluate each model
    all_results = {}

    for model_name in MODELS_TO_TRAIN:
        try:
            # Load trained model
            model = load_trained_model(model_name, DATASET_CONFIG["dataset_name"])

            # Evaluate
            result, y_pred_best, y_prob = evaluate_model(
                model, model_name, x_train, x_test, y_test,
                DATASET_CONFIG["dataset_name"]
            )

            # Save results
            save_evaluation_results(result, model_name, DATASET_CONFIG["dataset_name"])

            all_results[model_name] = result

        except FileNotFoundError as e:
            print(f"\nError: {e}")
            print("Skipping this model...")
            continue
        except Exception as e:
            print(f"\nUnexpected error with {model_name}: {e}")
            print("Skipping this model...")
            continue

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    for model_name, result in all_results.items():
        print(f"\n{model_name}:")
        print(f"ROC AUC: {result['metrics_at_optimal_threshold']['roc_auc']:.4f}")
        print(f"F1 Score (optimal threshold): {result['metrics_at_optimal_threshold']['f1_score']:.4f}")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)
    print(f"\nEvaluated {len(all_results)} model(s)")
    print(f"Results saved in: {PATHS['results_dir']}")

    return all_results


if __name__ == "__main__":
    results = run_evaluation()