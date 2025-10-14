import json
import os
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

from safeai_files.check_compliance import safeai_values
from safeai_files.check_explainability import compute_rge_values
from safeai_files.utils import save_model_metrics

# Data separation
data_lending_ny_clean = pd.read_csv("../saved_data/data_lending_clean_ny_original.csv")
predictors = data_lending_ny_clean.drop(columns=['action_taken'])
y = data_lending_ny_clean['action_taken']

# Split into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(predictors, y, test_size=0.2, random_state=15)
print("Training set shape:", x_train.shape)
print("Testing set shape:", x_test.shape)

# Random Classifier
random_model = DummyClassifier(strategy='prior', random_state=42)

# Train
random_model.fit(x_train, y_train)

# Make predictions
y_pred = random_model.predict(x_test)
y_prob = random_model.predict_proba(x_test)[:, 1]

# Integrating safeai
results = safeai_values(x_train, x_test, y_test, y_prob, random_model, "New York Original", "plots")
print(results)
save_model_metrics(results)

# Fairness
fair = compute_rge_values(x_train, x_test, y_prob, random_model, ["applicant_sex", "applicant_race_1"])
print(fair)

# Save results
data = {
    "x_final": results["x_final"],
    "y_final": results["y_final"],
    "z_final": results["z_final"]
}
json_str = json.dumps(data, indent=4)
file_path = os.path.join("../saved_data", "final_results_random_ny_original.json")
with open(file_path, "w", encoding="utf-8") as file:
    file.write(json_str)