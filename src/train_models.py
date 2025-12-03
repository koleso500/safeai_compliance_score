import joblib
import json
import optuna
import os
import pandas as pd
import logging
import time
import warnings
from tqdm.auto import tqdm
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

from src.config import DATASET_CONFIG, MODELS_TO_TRAIN, TRAINING_CONFIG, PATHS, MODELS_REQUIRING_SCALING
from src.models import MODEL_PARAM_FUNCTIONS, create_model, is_ensemble_model, get_required_base_models
from src import preprocessing

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)

def setup_optuna_file_logger():
    log_path = os.path.join(PATHS["results_dir"], "optuna_log.txt")

    # Create file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)

    # Format similar to Optuna's default layout
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)

    # Attach handler to optuna logger
    optuna_logger = optuna.logging.get_logger("optuna")
    optuna_logger.addHandler(file_handler)

    return log_path

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
    logger.info('Running preprocessing pipeline...')

    raw_path = os.path.join(PATHS['raw_data_dir'], DATASET_CONFIG['raw_data_file'])
    clean_path = preprocessing.preprocess(
        raw_data_path=raw_path,
        save_dir=PATHS['clean_data_dir'],
        dataset_name=dataset_name,
        create_sample=DATASET_CONFIG.get('create_sample', False),
        sample_fraction=DATASET_CONFIG.get('sample_fraction', 0.1)
    )

    logger.info(f'Loading clean data from: {clean_path}')
    data = pd.read_csv(clean_path)

    # Separate features and target
    x = data.drop(columns=[target_column])
    y = data[target_column]

    logger.info(f'Data shape: {data.shape}')
    logger.info(f'Features: {x.shape}')
    logger.info(f'Target: {y.shape}')

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
    logger.info('Splitting data...')

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info('Training set shape: %s', x_train.shape)
    logger.info('Testing set shape: %s', x_test.shape)

    # Save the splits
    os.makedirs(PATHS['clean_data_dir'], exist_ok=True)

    x_train.to_csv(os.path.join(PATHS['clean_data_dir'], f'x_train_{dataset_name}.csv'), index=False)
    x_test.to_csv(os.path.join(PATHS['clean_data_dir'], f'x_test_{dataset_name}.csv'), index=False)
    y_train.to_csv(os.path.join(PATHS['clean_data_dir'], f'y_train_{dataset_name}.csv'), index=False)
    y_test.to_csv(os.path.join(PATHS['clean_data_dir'], f'y_test_{dataset_name}.csv'), index=False)

    logger.info('Train/test splits saved to %s', PATHS['clean_data_dir'])

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
    logger.info('Scaling features...')

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
    os.makedirs(PATHS['models_dir'], exist_ok=True)
    scaler_path = os.path.join(PATHS['models_dir'], f'scaler_{dataset_name}.joblib')
    joblib.dump(scaler, scaler_path)
    logger.info(f'Scaler saved: {scaler_path}')

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
                    f'Cannot train {model_name}: required base models {missing}'
                    f'must be trained first. Current order: {models_to_train}'
                )
        trained.add(model_name)
    logger.info('Model training order validated successfully')


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
    logger.info(f'Optimizing: {model_name.upper()}')
    logger.info(f'Trials: {n_trials}, CV Folds: {cv_folds}, Metric: {scoring_metric}')

    # Define objective function
    def objective(trial):

        params = MODEL_PARAM_FUNCTIONS[model_name](trial)
        if model_name.lower() == "sem":
            if base_models is None:
                raise ValueError("SEM requires base_models")

            meta_train = {}
            for base_name, base_model in base_models.items():
                if hasattr(base_model, "predict_proba"):
                    meta_train[base_name + "_pred"] = base_model.predict_proba(x_train)[:, 1]
                else:
                    meta_train[base_name + "_pred"] = base_model.predict(x_train)

            meta_train = pd.DataFrame(meta_train)

            # Final estimator with trial params
            final_estimator = LogisticRegression(
                C=params["final_C"],
                penalty=params["final_penalty"],
                max_iter=params["final_max_iter"],
                class_weight=params.get("final_class_weight", None),
                solver="lbfgs",
                random_state=42,
            )

            # Evaluate final estimator using only meta-features
            scores = cross_val_score(
                final_estimator,
                meta_train,
                y_train,
                scoring=scoring_metric,
                cv=cv_folds,
                n_jobs=-1,
                error_score="raise"
            )

            return scores.mean()

        # For other models
        model = create_model(model_name, params, base_models)

        scores = cross_val_score(
            model,
            x_train,
            y_train,
            scoring=scoring_metric,
            cv=cv_folds,
            n_jobs=-1,
            error_score="raise"
        )
        return scores.mean()

    # Run optimization
    study = optuna.create_study(direction='maximize')

    # Add timeout and error handling
    try:
        start_time = time.time()
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        period = time.time() - start_time
        logger.info('Optimization finished in %.2f seconds', period)
    except Exception as opt_error:
        logger.error('Optimization failed for %s: %s', model_name, opt_error)
        raise

    best_params = study.best_trial.params
    logger.info('Best %s for %s: %.4f', scoring_metric, model_name, study.best_value)
    logger.info('Best parameters: %s', best_params)

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
    logger.info('Training final %s model with best parameters...', model_name)
    model = create_model(model_name, best_params, base_models=base_models)
    model.fit(x_train, y_train)
    logger.info('Training of %s complete', model_name)
    return model


def save_model_and_params(model, best_params, model_name, dataset_name):
    """
    Save trained model, parameters and optimization study

    Parameters
    ----------
    model: Trained model
    best_params: Best hyperparameters
    model_name: Name of the model
    dataset_name: Name of the dataset

    """
    logger.info('Saving model and results...')

    # Directories
    os.makedirs(PATHS['models_dir'], exist_ok=True)
    os.makedirs(PATHS['results_dir'], exist_ok=True)

    try:
        # Save model
        model_path = os.path.join(PATHS['models_dir'], f'{model_name}_{dataset_name}.joblib')
        joblib.dump(model, model_path)
        logger.info('Model saved to: %s', model_path)

        # Save parameters as JSON
        params_path = os.path.join(PATHS['results_dir'], f"{model_name}_{dataset_name}_params.json")
        with open(params_path, 'w', encoding='utf-8') as file:
            file.write(json.dumps(best_params, indent=4))
        logger.info('Parameters saved to: %s', params_path)

        logger.info('All results saved')

    except Exception as save_error:
        logger.error('Error saving model/results for %s: %s', model_name, save_error)
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
    logger.info('Loading base models for ensemble...')
    base_models = {}

    for model_name in base_model_names:
        model_path = os.path.join(PATHS['models_dir'], f'{model_name}_{dataset_name}.joblib')

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                 f"Base model not found: {model_path}. Train '{model_name}' before ensemble models."
            )

        base_models[model_name] = joblib.load(model_path)
        logger.info('Loaded base model: %s', model_name)

    return base_models


def grid_search_custom_params_f1(model_name: str, x_train, y_train, x_test, y_test, param_list, base_models=None):

    """
    Manual grid-search maximizing F1 on the test set with parallel execution using joblib.

    """
    logger.info(
        'Parallel F1 grid search for %s (%d configurations)',
        model_name,
        len(param_list)
    )

    def evaluate_config(config_id, params):
        """
        Evaluate a single configuration
        """
        warning_count = 0

        def warning_handler(_message, category, _filename, _lineno, _file=None, _line=None):
            nonlocal warning_count
            if issubclass(category, (
                    ConvergenceWarning,
                    RuntimeWarning,
                    UserWarning,
                    UndefinedMetricWarning
            )):
                warning_count += 1
            return

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('always')
                warnings.showwarning = warning_handler

                model = create_model(model_name, params, base_models)
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                f1 = f1_score(y_test, y_pred, average='binary')

            return {
                'config_id': config_id,
                'params': params,
                'f1': float(f1),
                'warnings': warning_count
            }

        except Exception as e:
            logger.error('Grid-search error (config %d): %s', config_id, e)
            return {
                'config_id': config_id,
                'params': params,
                'f1': None,
                'warnings': warning_count
            }

    # Build iterable of tasks
    tasks = (
        delayed(evaluate_config)(i + 1, params)
        for i, params in enumerate(param_list)
    )

    # Add tqdm progress bar ONLY for SVM (others stay clean)
    if model_name.lower() == "svm":
        tasks = tqdm(
            tasks,
            total=len(param_list),
            desc=f"SVM-grid-{model_name}",
            ncols=100
        )

    # Parallel execution
    results = Parallel(n_jobs=-1, backend='loky')(tasks)

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Drop None f1 values
    df_valid = df[df['f1'].notnull()].copy()

    if df_valid.empty:
        raise RuntimeError(f'No valid configuration found for {model_name}.')

    # Find best configuration
    idx_best = df_valid['f1'].idxmax()
    best_params = df_valid.loc[idx_best, 'params']
    best_f1 = df_valid.loc[idx_best, 'f1']

    # Aggregate warnings
    if 'warnings' in df.columns:
        total_warnings = int(df['warnings'].sum())
        logger.info(
            'Total warnings captured during grid-search for %s: %d',
            model_name,
            total_warnings
        )

    logger.info('Best F1 for %s = %.4f', model_name, best_f1)

    return best_params, float(best_f1), df


def run_training(use_custom_params=None):
    """
    Training pipeline supporting both:
    - Optuna optimization (default)
    - Custom parameter grids (direct F1 selection)

    Parameters
    ----------
    use_custom_params : dict or None
        Example:
        {
            "svm": [ {...}, {...}, ... ],
            "random_forest": {...}
        }

        If dict → fixed parameters
        If list → grid search over list (maximize F1 on test set)
    """
    log_path = setup_optuna_file_logger()

    ds_name = DATASET_CONFIG['dataset_name']
    logger.info('Training Pipeline')
    logger.info('Dataset: %s', ds_name)
    logger.info('Models to train: %s', MODELS_TO_TRAIN)
    logger.info('Custom grids enabled: %s', bool(use_custom_params))

    # Validate model order
    try:
        validate_model_order(MODELS_TO_TRAIN)
    except ValueError as e:
        logger.error('Model order validation failed: %s', e)
        raise

    # Load and preprocess
    try:
        x, y = load_clean_data(
            dataset_name=ds_name,
            target_column=DATASET_CONFIG['target_column']
        )
    except Exception as e:
        logger.error('Failed to load clean data: %s', e)
        raise

    # Train and test split
    try:
        x_train, x_test, y_train, y_test = split_and_save_data(
            x, y,
            dataset_name=ds_name,
            test_size=TRAINING_CONFIG['test_size'],
            random_state=TRAINING_CONFIG['random_state']
        )

        # Apply scaling if needed
        if any(m in MODELS_REQUIRING_SCALING for m in MODELS_TO_TRAIN):
            x_train_scaled, x_test_scaled, _ = scale_features(
                x_train, x_test, ds_name
            )
            logger.info('Scaled features available')
        else:
            x_train_scaled, x_test_scaled = x_train, x_test
            logger.info('Scaling not required')
    except Exception as e:
        logger.error('Split/scale step failed: %s', e)
        raise

    # Train Models
    models_trained = []
    failed_models = []

    for model_name in MODELS_TO_TRAIN:
        logger.info("\n" + "=" * 70 + "\n")
        logger.info('Processing Model: %s', model_name.upper())
        logger.info("-" * 70)

        try:
            # Choose scaled/unscaled data
            x_train_model = (x_train_scaled if model_name in MODELS_REQUIRING_SCALING else x_train)
            x_test_model = (x_test_scaled if model_name in MODELS_REQUIRING_SCALING else x_test)

            # Create local copies of y to prevent cross-contamination
            y_train_model = y_train.copy()
            y_test_model = y_test.copy()

            # QSVC-specific preprocessing
            if model_name == 'qsvc':
                logger.info('Applying QSVC-specific preprocessing...')

                # Reduce dataset size for QSVC
                if len(x_train_model) > 400:
                    x_train_model, _, y_train_model, _ = train_test_split(
                        x_train_model, y_train_model,
                        train_size=400,
                        stratify=y_train_model,
                        random_state=42
                    )
                    logger.info('Reduced QSVC training set to 400 samples')

                if len(x_test_model) > 400:
                    x_test_model, _, y_test_model, _ = train_test_split(
                        x_test_model, y_test_model,
                        train_size=400,
                        stratify=y_test_model,
                        random_state=42
                    )
                    logger.info('Reduced QSVC test set to 400 samples')

                # Apply PCA dimension reduction (ONLY for QSVC)
                pca = PCA(n_components=4, random_state=42)
                x_train_model = pca.fit_transform(x_train_model)
                x_test_model = pca.transform(x_test_model)
                logger.info('Applied PCA: reduced to 4 components for QSVC')

            logger.info('Using %s features',
                        'scaled' if model_name in MODELS_REQUIRING_SCALING else 'unscaled')
            logger.info('Training samples: %d, Test samples: %d', len(x_train_model), len(x_test_model))

            # Custom Parameter Grid Path
            if use_custom_params and model_name in use_custom_params:
                logger.info('Custom grid detected for %s, skipping Optuna', model_name)

                param_list = use_custom_params[model_name]

                # Load base models if ensemble (SEM, VEM)
                base_models = None
                if is_ensemble_model(model_name):
                    req = get_required_base_models(model_name)
                    base_models = load_base_models_for_ensemble(req, ds_name)

                # F1 maximization grid search
                best_params, best_f1, grid_df = grid_search_custom_params_f1(
                    model_name=model_name,
                    x_train=x_train_model,
                    y_train=y_train_model,
                    x_test=x_test_model,
                    y_test=y_test_model,
                    param_list=param_list,
                    base_models=base_models
                )

                # Save grid-search results (F1 scores for all parameter sets)
                grid_dir = os.path.join(PATHS['results_dir'], 'gridsearch')
                os.makedirs(grid_dir, exist_ok=True)

                # Sort by f1 descending
                grid_df_sorted = grid_df.sort_values(by='f1', ascending=False)

                csv_path = os.path.join(grid_dir, f'{model_name}_gridsearch_{ds_name}.csv')
                json_path = os.path.join(grid_dir, f'{model_name}_gridsearch_{ds_name}.json')

                grid_df_sorted.to_csv(csv_path, index=False)

                with open(json_path, 'w', encoding='utf-8') as f:
                    f.write(grid_df_sorted.to_json(orient='records', indent=4))

                logger.info('Saved sorted grid-search results for %s', model_name)
                logger.info('CSV : %s', csv_path)
                logger.info('JSON: %s', json_path)
                logger.info('Best F1 Score: %.4f', best_f1)

                # Train final model with best parameters
                model = train_final_model(
                    model_name,
                    best_params,
                    x_train_model,
                    y_train_model,
                    base_models=base_models
                )

                # Special handling for SEM (Stacking Ensemble Model)
                if model_name == 'sem':
                    logger.info('Generating meta-features for %s using pre-trained base models...', model_name.upper())

                    if base_models is None:
                        raise ValueError(f'{model_name.upper()} requires base_models')

                    # Generate meta-features from base model predictions
                    meta_train = {}
                    for base_name, base_model in base_models.items():
                        if hasattr(base_model, 'predict_proba'):
                            meta_train[base_name + '_pred'] = base_model.predict_proba(x_train_model)[:, 1]
                        else:
                            meta_train[base_name + '_pred'] = base_model.predict(x_train_model)

                    meta_train = pd.DataFrame(meta_train)

                    logger.info('Training final estimator of %s...', model_name.upper())
                    model.final_estimator_.fit(meta_train, y_train_model)

                else:
                    # Normal training for non-ensemble models
                    logger.info('Training %s with best parameters...', model_name.upper())
                    model.fit(x_train_model, y_train_model)

                model_tag = f'{model_name}_best'
                save_model_and_params(model, best_params, model_tag, ds_name)
                models_trained.append(model_tag)
                logger.info('Successfully trained and saved %s', model_tag)
                continue

            # Optuna Optimization Path
            else:
                logger.info('No custom grid for %s, running Optuna optimization', model_name)

                base_models = None
                if is_ensemble_model(model_name):
                    req = get_required_base_models(model_name)
                    base_models = load_base_models_for_ensemble(req, ds_name)

                # Run Optuna hyperparameter optimization
                best_params, _ = optimize_model(
                    model_name=model_name,
                    x_train=x_train_model,
                    y_train=y_train_model,
                    n_trials=TRAINING_CONFIG['n_trials'],
                    cv_folds=TRAINING_CONFIG['cv_folds'],
                    scoring_metric=TRAINING_CONFIG['scoring_metric'],
                    base_models=base_models
                )

                # Train final model with optimized parameters
                model = train_final_model(
                    model_name,
                    best_params,
                    x_train_model,
                    y_train_model,
                    base_models
                )

                # Special handling for SEM (Stacking Ensemble Model)
                if model_name == 'sem':
                    logger.info('Generating meta-features for %s using pre-trained base models...', model_name.upper())

                    if base_models is None:
                        raise ValueError(f'{model_name.upper()} requires base_models')

                    # Generate meta-features from base model predictions
                    meta_train = {}
                    for base_name, base_model in base_models.items():
                        if hasattr(base_model, 'predict_proba'):
                            meta_train[base_name + '_pred'] = base_model.predict_proba(x_train_model)[:, 1]
                        else:
                            meta_train[base_name + '_pred'] = base_model.predict(x_train_model)

                    meta_train = pd.DataFrame(meta_train)

                    logger.info('Training final estimator of %s...', model_name.upper())
                    model.final_estimator_.fit(meta_train, y_train_model)

                else:
                    # Normal training for non-ensemble models
                    logger.info('Training %s with optimized parameters...', model_name.upper())
                    model.fit(x_train_model, y_train_model)

                save_model_and_params(model, best_params, model_name, ds_name)
                models_trained.append(model_name)
                logger.info('Successfully trained and saved %s', model_name)

        except Exception as e:
            logger.error('Failed to train %s: %s', model_name, e)
            logger.error('Exception details:', exc_info=True)
            failed_models.append((model_name, str(e)))
            continue

    # Summary
    logger.info('Training Pipeline Complete')
    logger.info('Successfully trained models: %s', models_trained)

    if failed_models:
        logger.warning('\nModels that failed during training:')
        for m, err in failed_models:
            logger.warning('  - %s: %s', m, err)
    else:
        logger.info('\nAll models trained successfully')

    logger.info('\nOutput locations:')
    logger.info('Models saved in: %s', PATHS['models_dir'])
    logger.info('Parameters saved in: %s', PATHS['results_dir'])
    logger.info('Logs saved in: %s', log_path)

    return models_trained, failed_models


if __name__ == '__main__':
    try:
        trained_models, failed = run_training()

        if failed:
            logger.warning('Training finished with %d failure(s).', len(failed))
        else:
            logger.info('All models trained successfully')

    except Exception as pipeline_error:
        logger.error('Training pipeline failed: %s', pipeline_error)
        raise