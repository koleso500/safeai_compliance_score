import json
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, f1_score, roc_curve
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from safeai_files.check_compliance import safeai_values
from safeai_files.check_explainability import compute_rge_values
from safeai_files.utils import save_model_metrics

# Data separation
data_lending_ny_clean = pd.read_csv("../saved_data/data_lending_clean_ny_article.csv")
predictors = data_lending_ny_clean.drop(columns=['response'])
y = data_lending_ny_clean['response']

# Split into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(predictors, y, test_size=0.2, random_state=15)
print("Training set shape:", x_train.shape)
print("Testing set shape:", x_test.shape)

# Load best parameters for models
with open(os.path.join("../saved_data", "best_rf_params_ny_article.json"), "r", encoding="utf-8") as file_rf:
    best_rf_params = json.load(file_rf)

with open(os.path.join("../saved_data", "best_xgb_params_ny_article.json"), "r", encoding="utf-8") as file_xgb:
    best_xgb_params = json.load(file_xgb)

# Models
rf_clf = RandomForestClassifier(**best_rf_params, random_state=42, n_jobs=-1)
xgb_clf = XGBClassifier(**best_xgb_params, objective='binary:logistic', eval_metric='logloss')
lr_clf = LogisticRegression()

# Stacking Classifier with Logistic Regression as meta-estimator
stacking_clf = StackingClassifier(estimators=[('rf', rf_clf), ('xgb', xgb_clf)], final_estimator=lr_clf)

# Train
stacking_clf.fit(x_train, y_train)

# Make predictions
y_pred = stacking_clf.predict(x_test)
y_prob = stacking_clf.predict_proba(x_test)[:, 1]

#AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print("AUC:\n", roc_auc)

# Partial AUC
fpr_threshold = 0.3
valid_indices = np.where(fpr < fpr_threshold)[0]
partial_auc = auc(fpr[valid_indices], tpr[valid_indices]) / fpr_threshold
print(f"Partial AUC (FPR < {fpr_threshold}): {partial_auc:.4f}")

#ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc='lower right')
plt.savefig("plots/SEM_ROC_Curve_ny_article.png", dpi=300)
plt.close()

# DataFrame with thresholds, FPR, TPR, and F1 Score
roc_data = pd.DataFrame({
    'threshold': thresholds,
    'fpr': fpr,
    'tpr': tpr,
    'f1_score': [f1_score(y_test, np.where(y_prob >= t, 1, 0)) for t in thresholds]
})

# Best F1 score and corresponding threshold
best_f1 = roc_data.loc[roc_data['f1_score'].idxmax()]
best_threshold = best_f1["threshold"]
print(f"Best F1 Score: {best_f1["f1_score"]:.4f} at Threshold: {best_threshold:.4f}")

# Predictions at best threshold
y_pred_best = np.where(y_prob >= best_threshold, 1, 0)

# Evaluate Model Performance
print("Accuracy:", accuracy_score(y_test, y_pred_best))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best))
print("Classification Report:\n", classification_report(y_test, y_pred_best))

# Integrating safeai
results = safeai_values(x_train, x_test, y_test, y_prob, stacking_clf, "New York Article", "plots")
print(results)
save_model_metrics(results)

# Fairness
fair = compute_rge_values(x_train, x_test, y_prob, stacking_clf, ["applicant_sex_name", "applicant_race_1"])
print(fair)

# Save results
data = {
    "x_final": results["x_final"],
    "y_final": results["y_final"],
    "z_final": results["z_final"]
}
json_str = json.dumps(data, indent=4)
file_path = os.path.join("../saved_data", "final_results_stacked_ny_article.json")
with open(file_path, "w", encoding="utf-8") as file:
    file.write(json_str)