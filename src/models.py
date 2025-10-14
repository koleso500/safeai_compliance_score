from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import torch.nn as nn


def random_forest_params(trial):
    """
    Define hyperparameter search space for Random Forest

    Parameters
    ----------
    trial: trial object

    Returns
    -------
    Dictionary of hyperparameters

    """

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
    return params


def xgboost_params(trial):
    """
    Define hyperparameter search space for XGBoost

    Parameters
    ----------
    trial: trial object

    Returns
    -------
    Dictionary of hyperparameters

    """

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
    return params


def sem_params(trial):
    """
    Define hyperparameter search space for Stacked Ensemble Model

    Parameters
    ----------
    trial: trial object

    Returns
    -------
    Dictionary of hyperparameters

    """

    params = {
        'cv': trial.suggest_int('cv', 3, 10),
        'final_estimator_C': trial.suggest_float('final_estimator_C', 0.01, 10, log=True),
    }
    return params


def vem_params(trial):
    """
    Define hyperparameter search space for Voting Ensemble Model

    Parameters
    ----------
    trial: trial object

    Returns
    -------
    Dictionary of hyperparameters

    """

    params = {
        'voting': trial.suggest_categorical('voting', ['hard', 'soft']),
    }
    return params


class NeuralNetwork(nn.Module):
    """
    Custom Neural Network for binary classification

    """

    def __init__(self, input_size, hidden_layers, dropout_rate):

        super(NeuralNetwork, self).__init__()

        if isinstance(hidden_layers, int):
            hidden_layers = [hidden_layers]

        self.layers = nn.ModuleList()
        prev_size = input_size

        # Build hidden layers with ReLU and Dropout
        for size in hidden_layers:
            self.layers.append(nn.Linear(prev_size, size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
            prev_size = size

        # Output layer for binary classification
        self.layers.append(nn.Linear(prev_size, 1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def neural_network_params(trial):
    """
    Define hyperparameter search space for Neural Network

    Parameters
    ----------
    trial: trial object

    Returns
    -------
    Dictionary of hyperparameters

    """

    params = {
        'hidden_layers': trial.suggest_categorical('hidden_layers', [
            [64],
            [128, 64],
            [256, 128, 64],
            [128, 64, 32],
            [64, 32],
        ]),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.01, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
        'epochs': trial.suggest_int('epochs', 10, 100),
    }
    return params


def create_model(model_name, params, base_models=None):
    """
    Create a model instance with given parameters

    Parameters
    ----------
    model_name: Name of the model ("random_forest", "xgboost", "vem", "sem", "random", "neural_network")
    params: Dictionary of hyperparameters
    base_models: Dictionary of trained base models (required for VEM and SEM)

    Returns
    -------
    Untrained model instance

    """

    if model_name == "random_forest":
        return RandomForestClassifier(**params, random_state=42, n_jobs=-1)

    elif model_name == "xgboost":
        return XGBClassifier(**params, random_state=42, n_jobs=-1)

    elif model_name == "random":
        return DummyClassifier(strategy='prior', random_state=42)

    elif model_name == "sem":
        if base_models is None:
            raise ValueError("SEM requires base_models")

        # Create list of (name, model) tuples for StackingClassifier
        estimators = [(name, model) for name, model in base_models.items()]

        # Extract final estimator params
        final_estimator_c = params.pop('final_estimator_C', 1.0)
        final_estimator = LogisticRegression(C=final_estimator_c, random_state=42, max_iter=1000)

        return StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=params.get('cv', 5),
            n_jobs=-1
        )

    elif model_name == "vem":
        if base_models is None:
            raise ValueError("VEM requires base_models")
        # Create list of (name, model) tuples for VotingClassifier
        estimators = [(name, model) for name, model in base_models.items()]

        return VotingClassifier(
            estimators=estimators,
            voting=params.get('voting', 'soft'),
            n_jobs=-1
        )

    elif model_name == "neural_network":
        return {
            'type': 'neural_network',
            'params': params,
            'model_class': NeuralNetwork
        }

    else:
        raise ValueError(f"Unknown model: {model_name}")


def is_ensemble_model(model_name):

    """
    Check if a model is an ensemble that needs base models

    Parameters
    ----------
    model_name: Name of the model

    Returns
    -------
    Boolean - True if ensemble model

    """

    return model_name in ["vem", "sem"]


def get_required_base_models(model_name):
    """
    Get list of base models required for an ensemble

    Parameters
    ----------
    model_name: Name of the ensemble model

    Returns
    -------
    List of required base model names

    """

    if model_name == "vem" or model_name == "sem":
        return ["random_forest", "xgboost"]
    else:
        return []


# Dictionary mapping model names to their parameter functions
MODEL_PARAM_FUNCTIONS = {
    "random_forest": random_forest_params,
    "xgboost": xgboost_params,
    "vem": vem_params,
    "sem": sem_params,
}