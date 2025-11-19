import joblib
import json
import optuna
import os
import pandas as pd
import logging
import time
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import DATASET_CONFIG, MODELS_TO_TRAIN, TRAINING_CONFIG, PATHS, MODELS_REQUIRING_SCALING
from src.models import MODEL_PARAM_FUNCTIONS, create_model, is_ensemble_model, get_required_base_models
from src import preprocessing

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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

    # Always run preprocessing to ensure correct sample settings
    logger.info("Running preprocessing pipeline...")
    raw_path = os.path.join(PATHS["raw_data_dir"], DATASET_CONFIG["raw_data_file"])
    clean_path = preprocessing.preprocess(
        raw_data_path=raw_path,
        save_dir=PATHS["clean_data_dir"],
        dataset_name=dataset_name,
        create_sample=DATASET_CONFIG.get("create_sample", False),
        sample_fraction=DATASET_CONFIG.get("sample_fraction", 0.1)
    )

    logger.info(f"Loading clean data from: {clean_path}")
    data = pd.read_csv(clean_path)

    # Separate features and target
    x = data.drop(columns=[target_column])
    y = data[target_column]

    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Features: {x.shape}")
    logger.info(f"Target: {y.shape}")

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
        x, y, test_size=test_size, random_state=random_state, stratify=y
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


def scale_features(x_train, x_test, dataset_name):
    """
    Scale features using StandardScaler and save the scaler

    Parameters
    ----------
    x_train: Training features
    x_test: Testing features
    dataset_name: Name for saving scaler

    Returns
    -------
    Tuple of (x_train_scaled, x_test_scaled, scaler)
    """
    logger.info("\nScaling features...")

    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(
        scaler.fit_transform(x_train),
        columns=x_train.columns,
        index=x_train.index
    )
    x_test_scaled = pd.DataFrame(
        scaler.transform(x_test),
        columns=x_test.columns,
        index=x_test.index
    )

    # Save scaler
    scaler_path = os.path.join(PATHS["models_dir"], f"scaler_{dataset_name}.joblib")
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved: {scaler_path}")

    return x_train_scaled, x_test_scaled, scaler


def validate_model_order(models_to_train):
    """
    Validate that ensemble models come after their required base models

    Parameters
    ----------
    models_to_train: List of model names to train

    Raises
    ------
    ValueError: If ensemble models are listed before their required base models
    """
    trained = set()
    for model_name in models_to_train:
        if is_ensemble_model(model_name):
            required = get_required_base_models(model_name)
            missing = [m for m in required if m not in trained]
            if missing:
                raise ValueError(
                    f"Cannot train {model_name}: required base models {missing} "
                    f"must be trained first. Current order: {models_to_train}"
                )
        trained.add(model_name)
    logger.info("Model training order validated successfully")


def optimize_model(model_name, x_train, y_train, n_trials, cv_folds, scoring_metric, base_models=None):
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
    base_models: Dictionary of base models (required for ensemble models)

    Returns
    -------
    Tuple of (best_params, study)

    """

    logger.info("\n" + "=" * 60)
    logger.info(f"OPTIMIZING: {model_name.upper()}")
    logger.info("=" * 60)
    logger.info(f"Trials: {n_trials}, CV Folds: {cv_folds}, Metric: {scoring_metric}")

    # Get the parameter function for this model
    param_function = MODEL_PARAM_FUNCTIONS[model_name]

    # Define objective function
    def objective(trial):
        params = param_function(trial)
        model = create_model(model_name, params, base_models=base_models)

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

    # Add timeout and error handling
    try:
        start_time = time.time()
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        end_time = time.time()

        logger.info(f"\nOptimization completed in {end_time - start_time:.2f} seconds")
    except Exception as opt_error:
        logger.error(f"Optimization failed for {model_name}: {str(opt_error)}")
        raise

    best_params = study.best_trial.params
    logger.info(f"\nBest {scoring_metric} score: {study.best_value:.4f}")
    logger.info(f"Best parameters: {best_params}")

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


def save_model_and_params(model, best_params, study, model_name, dataset_name):
    """
    Save trained model, parameters, and optimization study

    Parameters
    ----------
    model: Trained model
    best_params: Best hyperparameters
    study: Optuna study object
    model_name: Name of the model
    dataset_name: Name of the dataset

    """

    logger.info("\nSaving model and results...")

    # Create directories
    os.makedirs(PATHS["models_dir"], exist_ok=True)
    os.makedirs(PATHS["results_dir"], exist_ok=True)

    try:
        # Save model
        model_path = os.path.join(PATHS["models_dir"], f"{model_name}_{dataset_name}.joblib")
        joblib.dump(model, model_path)
        logger.info(f"Model saved: {model_path}")

        # Save parameters as JSON
        params_path = os.path.join(PATHS["results_dir"], f"{model_name}_{dataset_name}_params.json")
        with open(params_path, 'w', encoding='utf-8') as file:
            file.write(json.dumps(best_params, indent=4))
        logger.info(f"Parameters saved: {params_path}")

        logger.info("All results saved!")

    except Exception as save_error:
        logger.error(f"Error saving model/results for {model_name}: {str(save_error)}")
        raise


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
    Training pipeline with comprehensive error handling

    """

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Dataset: {DATASET_CONFIG['dataset_name']}")
    logger.info(f"Models to train: {MODELS_TO_TRAIN}")
    logger.info("=" * 60)

    # Validate model training order
    try:
        validate_model_order(MODELS_TO_TRAIN)
    except ValueError as e:
        logger.error(f"Model order validation failed: {str(e)}")
        raise

    # Load clean data
    try:
        x, y = load_clean_data(
            dataset_name=DATASET_CONFIG["dataset_name"],
            target_column=DATASET_CONFIG["target_column"]
        )
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise

    # Split into train/test and save
    try:
        x_train, x_test, y_train, y_test = split_and_save_data(
            x, y,
            dataset_name=DATASET_CONFIG["dataset_name"],
            test_size=TRAINING_CONFIG["test_size"],
            random_state=TRAINING_CONFIG["random_state"]
        )

        # Apply scaling if any models require it
        if any(model in MODELS_REQUIRING_SCALING for model in MODELS_TO_TRAIN):
            x_train_scaled, x_test_scaled, scaler = scale_features(
                x_train, x_test, DATASET_CONFIG["dataset_name"]
            )
            logger.info("Features scaled for models requiring scaling")
        else:
            x_train_scaled, x_test_scaled = x_train, x_test
            logger.info("No scaling required for selected models")

    except Exception as e:
        logger.error(f"Failed to split/scale data: {str(e)}")
        raise

    # Train each selected model
    models_trained = []
    failed_models = []

    for model_name in MODELS_TO_TRAIN:
        logger.info("\n" + "=" * 60)
        logger.info(f"PROCESSING: {model_name.upper()}")
        logger.info("=" * 60)

        try:
            # Choose scaled or unscaled data based on model type
            x_train_model = x_train_scaled if model_name in MODELS_REQUIRING_SCALING else x_train

            if model_name in MODELS_REQUIRING_SCALING:
                logger.info(f"Using scaled features for {model_name}")
            else:
                logger.info(f"Using unscaled features for {model_name}")

            # Check if this is an ensemble model that needs base models
            if is_ensemble_model(model_name):
                required_base_models = get_required_base_models(model_name)
                logger.info(f"This is an ensemble model requiring: {required_base_models}")

                # Load base models
                base_models = load_base_models_for_ensemble(
                    required_base_models,
                    DATASET_CONFIG["dataset_name"]
                )

                # Optimize ensemble hyperparameters
                best_params, study = optimize_model(
                    model_name=model_name,
                    x_train=x_train_model,
                    y_train=y_train,
                    n_trials=TRAINING_CONFIG["n_trials"],
                    cv_folds=TRAINING_CONFIG["cv_folds"],
                    scoring_metric=TRAINING_CONFIG["scoring_metric"],
                    base_models=base_models
                )

                # Train final ensemble model
                model = train_final_model(
                    model_name, best_params, x_train_model, y_train, base_models=base_models
                )
            else:
                # Regular model
                best_params, study = optimize_model(
                    model_name=model_name,
                    x_train=x_train_model,
                    y_train=y_train,
                    n_trials=TRAINING_CONFIG["n_trials"],
                    cv_folds=TRAINING_CONFIG["cv_folds"],
                    scoring_metric=TRAINING_CONFIG["scoring_metric"]
                )

                # Train final model
                model = train_final_model(model_name, best_params, x_train_model, y_train)

            # Save model and parameters
            save_model_and_params(
                model, best_params, study, model_name, DATASET_CONFIG["dataset_name"]
            )

            models_trained.append(model_name)
            logger.info(f"✓ Successfully trained {model_name}")

        except Exception as e:
            logger.error(f"✗ Failed to train {model_name}: {str(e)}")
            failed_models.append((model_name, str(e)))
            # Continue with next model instead of stopping
            continue

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"\nSuccessfully trained models: {models_trained}")

    if failed_models:
        logger.warning(f"\nFailed models:")
        for model_name, error in failed_models:
            logger.warning(f"  - {model_name}: {error}")

    logger.info(f"\nModels saved in: {PATHS['models_dir']}")
    logger.info(f"Parameters saved in: {PATHS['results_dir']}")

    return models_trained, failed_models


if __name__ == "__main__":
    try:
        trained_models, failed_models = run_training()

        if failed_models:
            logger.warning(f"\nTraining completed with {len(failed_models)} failure(s)")
        else:
            logger.info("\nAll models trained successfully!")

    except Exception as pipeline_error:
        logger.error(f"Training pipeline failed: {str(pipeline_error)}")
        raise