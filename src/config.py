import os

# Directories Configuration
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CONFIG_DIR)

PATHS = {
    'raw_data_dir': os.path.join(PROJECT_ROOT, 'data', 'raw'),
    'clean_data_dir': os.path.join(PROJECT_ROOT, 'data', 'processed'),
    'models_dir': os.path.join(PROJECT_ROOT, 'models'),
    'results_dir': os.path.join(PROJECT_ROOT, 'results')
}

# Dataset Configuration
DATASET_CONFIG = {
    'dataset_name': 'ny_2017',  # for saving files
    'raw_data_file': 'hmda_2017_ny_all-records_labels.csv',
    'target_column': 'action_taken',
    'create_sample': True,
    'sample_fraction': 0.1
}

# Models Configuration
MODELS_TO_TRAIN = [
    'logistic_regression',
    'random_forest',
    'xgboost',
    'vem',
    'sem',
    'svm',
    'neural_network',
    'qsvc'
]

# Models that require feature scaling
MODELS_REQUIRING_SCALING = [
    'logistic_regression',
    'svm',
    'neural_network',
    'qsvc'
]

# Training Configuration
TRAINING_CONFIG = {
    'test_size': 0.2,
    'random_state': 15,
    'cv_folds': 3,
    'n_trials': 10, # Optuna optimization trials
    'scoring_metric': 'f1'
}

# TOPSIS weights for each dimension (must sum to 1.0)
TOPSIS_WEIGHTS = [1/3, 1/3, 1/3]  # Equal weights by default