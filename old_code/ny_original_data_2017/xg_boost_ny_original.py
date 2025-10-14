import joblib
import json
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice, plot_parallel_coordinate
import os
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
import xgboost as xgb

# Data separation
data_lending_ny_clean = pd.read_csv("../saved_data/data_lending_clean_ny_original.csv")
x = data_lending_ny_clean.drop(columns=['action_taken'])
y = data_lending_ny_clean['action_taken']

# Split into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=15)
print("Training set shape:", x_train.shape)
print("Testing set shape:", x_test.shape)

# Save the splits
x_train.to_csv(os.path.join("../saved_data", "x_train_xgb_ny_original.csv"), index=False)
x_test.to_csv(os.path.join("../saved_data", "x_test_xgb_ny_original.csv"), index=False)
y_train.to_csv(os.path.join("../saved_data", "y_train_xgb_ny_original.csv"), index=False)
y_test.to_csv(os.path.join("../saved_data", "y_test_xgb_ny_original.csv"), index=False)

# XGBoost classifier with Optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 10),
        'random_state': 42,
    }

    xgb_model = xgb.XGBClassifier(**params, objective='binary:logistic', eval_metric='logloss')
    score = cross_val_score(xgb_model, x_train, y_train, scoring='f1', cv=5).mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Best model, parameters and F1
best_params = study.best_trial.params
best_model = xgb.XGBClassifier(**best_params, objective='binary:logistic', eval_metric='logloss')
best_model.fit(x_train, y_train)
print("Best Parameters:", best_params)

# Save best parameters and model
json_str = json.dumps(best_params, indent=4)
file_path = os.path.join("../saved_data", "best_xgb_params_ny_original.json")
with open(file_path, "w", encoding="utf-8") as file:
    file.write(json_str)
print("Best parameters saved successfully!")

model_path = os.path.join("../saved_models", "best_xgb_model_ny_original.joblib")
joblib.dump(best_model, model_path)

# Some plots
plot_optimization_history(study).show()
plot_param_importances(study).show()
plot_parallel_coordinate(study).show()
plot_slice(study).show()