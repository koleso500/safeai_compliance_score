import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, f1_score, roc_curve
import torch

from safeai_files.check_compliance import safeai_values
from safeai_files.check_explainability import compute_rge_values
from safeai_files.utils import save_model_metrics
from old_code.torch_for_credits.torch_model import NeuralNetwork

# Directory of the data
save_dir = "../saved_data"

# Load torch tensors
tensor_path = os.path.join(save_dir, "full_data_tensors_ny_article.pth")
tensor_data = torch.load(tensor_path, weights_only=True)
x_train_tensor = tensor_data["x_train_tensor_ny_article"]
y_train_tensor = tensor_data["y_train_tensor_ny_article"]
x_test_tensor = tensor_data["x_test_tensor_ny_article"]
y_test_tensor = tensor_data["y_test_tensor_ny_article"]

# Load best torch parameters
best_params_path = os.path.join(save_dir, "best_torch_params_ny_article.json")
with open(best_params_path, "r", encoding="utf-8") as file:
    best_params = json.load(file)

# Load other data
x_train_scaled_names = pd.read_csv(os.path.join(save_dir, "x_train_scaled_names_ny_article.csv"))
x_test_scaled_names = pd.read_csv(os.path.join(save_dir, "x_test_scaled_names_ny_article.csv"))
y_train = np.load(os.path.join(save_dir, "y_train_ny_article.npy"))
y_test = np.load(os.path.join(save_dir, "y_test_ny_article.npy"))

# Load loss values
train_losses = np.load("../saved_data/best_train_losses_ny_article.npy")
val_losses = np.load("../saved_data/best_val_losses_ny_article.npy")

def evaluate_model_ny_article():
    # Load the best model
    best_model = NeuralNetwork(x_train_scaled_names.shape[1], best_params[2], best_params[3])
    best_model.load_state_dict(torch.load('../saved_models/best_torch_model_ny_article.pth', weights_only=True))
    best_model.eval()

    # Make predictions on the test set
    torch.manual_seed(42)
    with torch.no_grad():
        y_pred = best_model(x_test_tensor)

        # Apply sigmoid to convert to probabilities if needed
        if y_pred.shape[1] == 1:
            y_pred_prob = torch.sigmoid(y_pred).cpu().numpy()
        else:
            y_pred_prob = y_pred.cpu().numpy()

    # Convert to numpy for evaluation
    y_test_numpy = y_test_tensor.cpu().numpy()

    # Flatten
    y_pred_prob_flat = y_pred_prob.flatten()

    # Search for the best threshold based on F1-score
    best_f1 = 0
    best_threshold = 0

    for threshold in np.arange(0.0, 1.01, 0.01):
        y_pred_labels = np.where(y_pred_prob_flat > threshold, 1, 0)
        f1 = f1_score(y_test_numpy, y_pred_labels)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"Best threshold based on F1-score: {best_threshold:.2f} (F1: {best_f1:.4f})")

    # Apply the best threshold
    y_pred_labels = np.where(y_pred_prob_flat > best_threshold, 1, 0)

    # Evaluate model performance
    print("Accuracy:", accuracy_score(y_test_numpy, y_pred_labels))
    print("Confusion Matrix:\n", confusion_matrix(y_test_numpy, y_pred_labels))
    print("Classification Report:\n", classification_report(y_test_numpy, y_pred_labels))

    # AUC
    fpr, tpr, thresholds = roc_curve(y_test_numpy, y_pred_prob_flat)
    roc_auc = auc(fpr, tpr)
    print("AUC Score:", roc_auc)

    # Partial AUC
    fpr_threshold = 0.3
    valid_indices = np.where(fpr < fpr_threshold)[0]
    partial_auc = auc(fpr[valid_indices], tpr[valid_indices]) / fpr_threshold
    print(f"Partial AUC (FPR < {fpr_threshold}): {partial_auc:.4f}")

    # ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc='lower right')
    plt.savefig("plots/NN_ROC_Curve_ny_article.png", dpi=300)
    plt.close()

    # Training and Validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss", linewidth=2)
    plt.plot(val_losses, label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss (Best Fold)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/NN_Loss_ny_article.png", dpi=300)
    plt.close()

    # Integrating safeai
    results = safeai_values(x_train_scaled_names, x_test_scaled_names, y_test, y_pred_prob, best_model,
                            "New York Article", "plots")
    print(results)
    save_model_metrics(results)

    # Fairness
    fair = compute_rge_values(x_train_scaled_names, x_test_scaled_names, y_pred_prob, best_model,["applicant_sex_name", "applicant_race_1"])
    print(fair)

    # Save results
    data = {
        "x_final": results["x_final"],
        "y_final": results["y_final"],
        "z_final": results["z_final"]
    }
    json_str = json.dumps(data, indent=4)
    file_path = os.path.join("../saved_data", "final_results_neural_ny_article.json")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(json_str)

if __name__ == "__main__":
    evaluate_model_ny_article()