import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.circuit.library import PauliFeatureMap


def logistic_regression_params(trial):
    """
    Define hyperparameter search space for Logistic Regression

    Parameters
    ----------
    trial: trial object

    Returns
    -------
    Dictionary of hyperparameters

    """
    params = {
        'C': trial.suggest_float('C', 0.001, 100, log=True),
        'penalty': trial.suggest_categorical('penalty', ['l2']),
        'max_iter': 1000,
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
        'solver': 'lbfgs',
    }

    return params


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
        'max_features': trial.suggest_categorical(
            'max_features',
            ['sqrt', 'log2', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        ),
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
        'final_C': trial.suggest_float('final_C', 0.001, 100, log=True),
        'final_penalty': trial.suggest_categorical('final_penalty', ['l2']),
        'final_max_iter': trial.suggest_int('final_max_iter', 100, 1000),
        'final_class_weight': trial.suggest_categorical('final_class_weight', [None, 'balanced']),
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


def svm_params(trial):
    """
    Define hyperparameter search space for Support Vector Machine

    Parameters
    ----------
    trial: trial object

    Returns
    -------
    Dictionary of hyperparameters

    """
    params = {
        'C': trial.suggest_float('C', 0.001, 100, log=True),
        'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid']),
        'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
        'degree': trial.suggest_int('degree', 2, 5),  # for poly kernel
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
        'probability': True, # always true for predict_proba
        'max_iter': 50000,
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


class NeuralNetworkClassifier(BaseEstimator, ClassifierMixin):
    """

    """

    def __init__(self, hidden_layers=None, dropout_rate=0.3, learning_rate=0.001,
                 batch_size=32, epochs=50, random_state=42, verbose=0):
        self.hidden_layers = hidden_layers if hidden_layers is not None else [64]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        self.verbose = verbose
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes_ = None

    def fit(self, x, y):
        """
        Train the neural network

        Parameters
        ----------
        x: Training features
        y: Training labels

        Returns
        -------
        self

        """
        # Set random seeds
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Store classes
        self.classes_ = np.unique(y)

        # Convert to numpy if needed
        if hasattr(x, 'values'):
            x = x.values
        if hasattr(y, 'values'):
            y = y.values

        # Convert to tensors
        x_tensor = torch.FloatTensor(x).to(self.device)
        y_tensor = torch.FloatTensor(y.reshape(-1, 1)).to(self.device)

        # Create dataset and dataloader
        dataset = TensorDataset(x_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize model
        input_size = x.shape[1]
        self.model = NeuralNetwork(input_size, self.hidden_layers, self.dropout_rate).to(self.device)

        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_x, batch_y in dataloader:
                # Forward pass
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if self.verbose > 0 and (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / len(dataloader)
                print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_loss:.4f}")

        return self

    def predict_proba(self, x):
        """
        Predict class probabilities

        Parameters
        ----------
        x: Features

        Returns
        -------
        Array of probabilities for each class

        """

        if self.model is None:
            raise RuntimeError('Model is not fitted yet')

        # Convert to numpy if needed
        if hasattr(x, 'values'):
            x = x.values

        x_tensor = torch.FloatTensor(x).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()

        # Return probabilities for both classes
        return np.hstack([1 - probs, probs])

    def predict(self, x):
        """
        Predict class labels

        Parameters
        ----------
        x: Features

        Returns
        -------
        Array of predicted class labels

        """
        probs = self.predict_proba(x)
        return (probs[:, 1] >= 0.5).astype(int)

    def score(self, x, y, sample_weight=None):
        """
        Return F1 score

        Parameters
        ----------
        x: Features
        y: True labels
        sample_weight: Sample weights (optional)

        Returns
        -------
        F1 score

        """
        return f1_score(y, self.predict(x), sample_weight=sample_weight)


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


class QSVCWrapper(BaseEstimator, ClassifierMixin):

    """
    Sklearn-style QSVC to treat it like a normal classifier

    """
    def __init__(self, c=1.0, tol=1e-3, max_iter=-1, feature_dim=4):
        self.c = c
        self.tol = tol
        self.max_iter = max_iter
        self.feature_dim = feature_dim

        self.pca = PCA(n_components=feature_dim, random_state=42)

        feature_map = PauliFeatureMap(
            feature_dimension=feature_dim,
            reps=1,
            paulis=['Z', 'ZZ'],
        )

        quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

        self.qsvc_ = QSVC(
            quantum_kernel=quantum_kernel,
            C=c,
            tol=tol,
            max_iter=max_iter
        )

    @staticmethod
    def _to_array(x):
        if hasattr(x, "values"):
            return x.values
        return np.asarray(x)

    def fit(self, x, y):
        x_arr = self._to_array(x)
        x_reduced = self.pca.fit_transform(x_arr)
        self.qsvc_.fit(x_reduced, y)
        return self

    def predict(self, x):
        x_arr = self._to_array(x)
        x_reduced = self.pca.transform(x_arr)
        return self.qsvc_.predict(x_reduced)

    def predict_proba(self, x):
        x_arr = self._to_array(x)
        x_reduced = self.pca.transform(x_arr)
        try:
            return self.qsvc_.predict_proba(x_reduced)
        except Exception:
            scores = self.qsvc_.decision_function(x_reduced)

            # Min-max scale to [0,1]
            min_s = scores.min()
            max_s = scores.max()
            if max_s - min_s < 1e-12:
                probs_pos = np.zeros_like(scores)
            else:
                probs_pos = (scores - min_s) / (max_s - min_s)

            probs_neg = 1 - probs_pos
            return np.column_stack((probs_neg, probs_pos))


def qsvc_params(trial):
    """
    Define hyperparameter search space for Quantum SVM (QSVC).

    Parameters
    ----------
    trial: trial object

    Returns
    -------
    Dictionary of hyperparameters

    """
    params =  {
        "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
        "tol": trial.suggest_float("tol", 1e-5, 1e-2, log=True),
        "max_iter": trial.suggest_int("max_iter", 1000, 5000),
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
    params_local = dict(params)

    if model_name == 'logistic_regression':
        return LogisticRegression(**params_local, random_state=42, n_jobs=-1)

    elif model_name == 'random_forest':
        return RandomForestClassifier(**params_local, random_state=42, n_jobs=-1)

    elif model_name == 'xgboost':
        return XGBClassifier(**params_local, random_state=42, n_jobs=-1)

    elif model_name == 'sem':
        if base_models is None:
            raise ValueError('SEM requires base_models')

        # Base estimators
        estimators = [(name, model) for name, model in base_models.items()]

        # Extract and create final estimator
        final_estimator_params = {
            'C': params_local.pop('final_C', 1.0),
            'penalty': params_local.pop('final_penalty', 'l2'),
            'max_iter': params_local.pop('final_max_iter', 1000),
            'class_weight': params_local.pop('final_class_weight', None),
            'solver': 'lbfgs',
            'random_state': 42,
        }

        final_estimator = LogisticRegression(**final_estimator_params)

        return StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv='prefit',
            stack_method='auto',
            n_jobs=-1
        )

    elif model_name == 'vem':
        if base_models is None:
            raise ValueError('VEM requires base_models')

        # Base estimators
        estimators = [(name, model) for name, model in base_models.items()]

        return VotingClassifier(
            estimators=estimators,
            voting=params_local.get('voting', 'soft'),
            n_jobs=-1
        )

    elif model_name == 'svm':
        # Ensure probability is always True for SVM
        params_local['probability'] = True
        return SVC(**params_local, random_state=42)

    elif model_name == 'neural_network':
        return NeuralNetworkClassifier(**params_local, random_state=42)

    elif model_name == 'random':
        return DummyClassifier(strategy='prior', random_state=42)

    elif model_name == "qsvc":
        c = params_local.pop("C", 1.0)
        tol = params_local.pop("tol", 1e-3)
        max_iter = params_local.pop("max_iter", -1)

        return QSVCWrapper(
            c=c,
            tol=tol,
            max_iter=max_iter,
        )

    else:
        raise ValueError(f'Unknown model: {model_name}')


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

    return model_name in ['vem', 'sem']


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

    if model_name == 'vem' or model_name == 'sem':
        return ['random_forest', 'xgboost']
    else:
        return []


# Dictionary mapping model names to their parameter functions
MODEL_PARAM_FUNCTIONS = {
    'logistic_regression': logistic_regression_params,
    'random_forest': random_forest_params,
    'xgboost': xgboost_params,
    'vem': vem_params,
    'sem': sem_params,
    'svm': svm_params,
    'neural_network': neural_network_params,
    "qsvc": qsvc_params
}