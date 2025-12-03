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

# TOPSIS Configuration

# Ideal values for TOPSIS ranking (best case scenario)
TOPSIS_IDEAL_VALUES = {
    'x': [1.0] * 21,    # Perfect RGA
    'y': [1.0] + [0.0] * 20,    # Perfect RGE
    'z': [1.0] * 21,    # Perfect RGR
}

# Worst values for TOPSIS ranking (worst case scenario)
TOPSIS_WORST_VALUES = {
    'x': [0.0] * 21,    # Worst RGA
    'y': [
        1.0,
        0.9736842105263158,
        0.9473684210526316,
        0.9210526315789473,
        0.8947368421052632,
        0.868421052631579,
        0.8421052631578947,
        0.8157894736842105,
        0.7894736842105263,
        0.7631578947368421,
        0.736842105263158,
        0.7105263157894737,
        0.6842105263157895,
        0.6578947368421053,
        0.631578947368421,
        0.6052631578947368,
        0.5789473684210527,
        0.5526315789473685,
        0.5263157894736843,
        0.5163157894736843,
        0.5,
    ],  # Worst RGE
    'z': [1.0] + [0.0] * 20,  # Worst RGR
}

# TOPSIS weights for each dimension (must sum to 1.0)
TOPSIS_WEIGHTS = [1/3, 1/3, 1/3]  # Equal weights by default