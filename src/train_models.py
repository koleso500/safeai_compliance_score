import joblib
import json
import optuna
import os
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split

from src.config import DATASET_CONFIG, MODELS_TO_TRAIN, TRAINING_CONFIG, PATHS
from src.models import MODEL_PARAM_FUNCTIONS, create_model, is_ensemble_model, get_required_base_models
from src import preprocessing


def load_clean_data(dataset_name, target_column):

    """
    Load cleaned data

    Parameters
    ----------
    dataset_name: Name of the dataset
    target_column: Name of target variable

    Returns
    -------
    Tuple of (x, y) - features and target

    """

    clean_path = os.path.join(PATHS["clean_data_dir"], f"data_lending_clean_{dataset_name}.csv")

    if not os.path.exists(clean_path):
        print(f"Clean data not found at: {clean_path}")
        print("Running preprocessing pipeline...")
        raw_path = os.path.join(PATHS["raw_data_dir"], DATASET_CONFIG["raw_data_file"])
        clean_path = preprocessing.preprocess(
            raw_data_path=raw_path,
            save_dir=PATHS["clean_data_dir"],
            dataset_name=dataset_name
        )

    print(f"\nLoading clean data from: {clean_path}")
    data = pd.read_csv(clean_path)

    # Separate features and target
    x = data.drop(columns=[target_column])
    y = data[target_column]

    print(f"Data shape: {data.shape}")
    print(f"Features: {x.shape}")
    print(f"Target: {y.shape}")

    return x, y


def split_and_save_data(x, y, dataset_name, test_size, random_state):

    """
    Split data into train/test and save splits

    Parameters
    ----------
    x: Features
    y: Target
    dataset_name: Name for saving files
    test_size: Proportion for test set
    random_state: Random seed

    Returns
    -------
    Tuple of (x_train, x_test, y_train, y_test)

    """

    print("\n" + "=" * 60)
    print("SPLITTING DATA")
    print("=" * 60)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    print(f"Training set shape: {x_train.shape}")
    print(f"Testing set shape: {x_test.shape}")

    # Save the splits
    os.makedirs(PATHS["clean_data_dir"], exist_ok=True)
    x_train.to_csv(os.path.join(PATHS["clean_data_dir"], f"x_train_{dataset_name}.csv"), index=False)
    x_test.to_csv(os.path.join(PATHS["clean_data_dir"], f"x_test_{dataset_name}.csv"), index=False)
    y_train.to_csv(os.path.join(PATHS["clean_data_dir"], f"y_train_{dataset_name}.csv"), index=False)
    y_test.to_csv(os.path.join(PATHS["clean_data_dir"], f"y_test_{dataset_name}.csv"), index=False)
    print("Train/test splits saved!")

    return x_train, x_test, y_train, y_test


def optimize_model(model_name, x_train, y_train, n_trials, cv_folds, scoring_metric):

    """
    Run Optuna optimization for model hyperparameters

    Parameters
    ----------
    model_name: Name of the model to optimize
    x_train: Training features
    y_train: Training target
    n_trials: Number of Optuna trials
    cv_folds: Number of cross-validation folds
    scoring_metric: Metric to optimize (e.g., 'f1')

    Returns
    -------
    Tuple of (best_params, study)

    """

    print("\n" + "=" * 60)
    print(f"OPTIMIZING: {model_name.upper()}")
    print("=" * 60)
    print(f"Trials: {n_trials}, CV Folds: {cv_folds}, Metric: {scoring_metric}")

    # Get the parameter function for this model
    param_function = MODEL_PARAM_FUNCTIONS[model_name]

    # Define objective function
    def objective(trial):
        params = param_function(trial)
        model = create_model(model_name, params)

        # Evaluate
        score = cross_val_score(
            model, x_train, y_train,
            scoring=scoring_metric,
            cv=cv_folds,
            n_jobs=-1
        ).mean()

        return score

    # Run optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_trial.params
    print(f"\nBest {scoring_metric} score: {study.best_value:.4f}")
    print(f"Best parameters: {best_params}")

    return best_params, study


def train_final_model(model_name, best_params, x_train, y_train, base_models=None):

    """
    Train final model with best parameters

    Parameters
    ----------
    model_name: Name of the model
    best_params: Best hyperparameters from Optuna
    x_train: Training features
    y_train: Training target
    base_models: Dictionary of base models (for ensembles)

    Returns
    -------
    Trained model

    """

    print(f"\nTraining final {model_name} model...")
    model = create_model(model_name, best_params, base_models=base_models)
    model.fit(x_train, y_train)
    print("Training complete!")
    return model


def save_model_and_params(model, best_params, model_name, dataset_name):

    """
    Save trained model and parameters

    Parameters
    ----------
    model: Trained model
    best_params: Best hyperparameters
    model_name: Name of the model
    dataset_name: Name of the dataset

    Returns
    -------

    """

    print("\nSaving model and results...")

    # Create directories
    os.makedirs(PATHS["models_dir"], exist_ok=True)
    os.makedirs(PATHS["results_dir"], exist_ok=True)

    # Save model
    model_path = os.path.join(PATHS["models_dir"], f"{model_name}_{dataset_name}.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved: {model_path}")

    # Save parameters as JSON
    json_str = json.dumps(best_params, indent=4)
    params_path = os.path.join(PATHS["results_dir"], f"{model_name}_{dataset_name}_params.json")
    with open(params_path, 'w', encoding='utf-8') as file:
        file.write(json_str)
    print(f"Parameters saved: {params_path}")
    print("All results saved!")


def load_base_models_for_ensemble(base_model_names, dataset_name):

    """
    Load previously trained base models for ensemble methods

    Parameters
    ----------
    base_model_names: List of base model names to load
    dataset_name: Name of the dataset

    Returns
    -------
    Dictionary of {model_name: trained_model}

    """

    print("\nLoading base models for ensemble...")
    base_models = {}

    for model_name in base_model_names:
        model_path = os.path.join(PATHS["models_dir"], f"{model_name}_{dataset_name}.joblib")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Base model not found: {model_path}\n"
                f"Train {model_name} first before training ensemble models."
            )

        base_models[model_name] = joblib.load(model_path)
        print(f"Loaded: {model_name}")

    return base_models


def run_training():
    """
    Training pipeline

    """

    print("\n" + "=" * 60)
    print("TRAINING PIPELINE")
    print("=" * 60)
    print(f"Dataset: {DATASET_CONFIG['dataset_name']}")
    print(f"Models to train: {MODELS_TO_TRAIN}")
    print("=" * 60)

    # Load clean data
    x, y = load_clean_data(
        dataset_name=DATASET_CONFIG["dataset_name"],
        target_column=DATASET_CONFIG["target_column"]
    )

    # Split into train/test and save
    x_train, x_test, y_train, y_test = split_and_save_data(
        x, y,
        dataset_name=DATASET_CONFIG["dataset_name"],
        test_size=TRAINING_CONFIG["test_size"],
        random_state=TRAINING_CONFIG["random_state"]
    )

    # Train each selected model
    models_trained = []

    for model_name in MODELS_TO_TRAIN:
        print("\n" + "=" * 60)
        print(f"PROCESSING: {model_name.upper()}")
        print("=" * 60)

        # Check if this is an ensemble model that needs base models
        if is_ensemble_model(model_name):
            required_base_models = get_required_base_models(model_name)
            print(f"This is an ensemble model requiring: {required_base_models}")

            # Load base models
            base_models = load_base_models_for_ensemble(
                required_base_models,
                DATASET_CONFIG["dataset_name"]
            )

            # Optimize ensemble hyperparameters
            best_params, study = optimize_model(
                model_name=model_name,
                x_train=x_train,
                y_train=y_train,
                n_trials=TRAINING_CONFIG["n_trials"],
                cv_folds=TRAINING_CONFIG["cv_folds"],
                scoring_metric=TRAINING_CONFIG["scoring_metric"]
            )

            # Train final ensemble model
            model = train_final_model(
                model_name, best_params, x_train, y_train, base_models=base_models
            )
        else:
            # Regular model
            best_params, study = optimize_model(
                model_name=model_name,
                x_train=x_train,
                y_train=y_train,
                n_trials=TRAINING_CONFIG["n_trials"],
                cv_folds=TRAINING_CONFIG["cv_folds"],
                scoring_metric=TRAINING_CONFIG["scoring_metric"]
            )

            # Train final model
            model = train_final_model(model_name, best_params, x_train, y_train)

        # Save model and parameters
        save_model_and_params(
            model, best_params, model_name, DATASET_CONFIG["dataset_name"]
        )

        models_trained.append(model_name)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nTrained models: {models_trained}")
    print(f"\nModels saved in: {PATHS['models_dir']}")
    print(f"Parameters saved in: {PATHS['results_dir']}")

    return models_trained


if __name__ == "__main__":
    trained_models = run_training()