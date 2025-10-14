import joblib
import json
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice, plot_parallel_coordinate
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split

# Data separation
data_lending_ny_clean = pd.read_csv("../saved_data/data_lending_clean_ny_article.csv")
x = data_lending_ny_clean.drop(columns=['response'])
y = data_lending_ny_clean['response']

# Split into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=15)
print("Training set shape:", x_train.shape)
print("Testing set shape:", x_test.shape)

# Save the splits
x_train.to_csv(os.path.join("../saved_data", "x_train_rf_ny_article.csv"), index=False)
x_test.to_csv(os.path.join("../saved_data", "x_test_rf_ny_article.csv"), index=False)
y_train.to_csv(os.path.join("../saved_data", "y_train_rf_ny_article.csv"), index=False)
y_test.to_csv(os.path.join("../saved_data", "y_test_rf_ny_article.csv"), index=False)

# Random Forest with Optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_categorical('max_depth', [None, 5, 10, 15, 20]),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                                                                   0.8, 0.9, 1]),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 1000),
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced', 'balanced_subsample']),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
    }

    rf_model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    score = cross_val_score(rf_model, x_train, y_train, scoring='f1', cv=5).mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Best model, parameters and F1
best_params = study.best_trial.params
best_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
best_model.fit(x_train, y_train)
print("Best Parameters:", best_params)

# Save best parameters and model
json_str = json.dumps(best_params, indent=4)
file_path = os.path.join("../saved_data", "best_rf_params_ny_article.json")
with open(file_path, "w", encoding="utf-8") as file:
    file.write(json_str)
print("Best parameters saved successfully!")

model_path = os.path.join("../saved_models", "best_rf_model_ny_article.joblib")
joblib.dump(best_model, model_path)

#  Some plots
plot_optimization_history(study).show()
plot_param_importances(study).show()
plot_parallel_coordinate(study).show()
plot_slice(study).show()