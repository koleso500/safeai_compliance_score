import logging
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from src.config import DATASET_CONFIG, PATHS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_raw_data(file_path):
    """
    Load raw data from CSV file

    Parameters
    ----------
    file_path: Path to the CSV file

    Returns
    -------
    pandas DataFrame with raw data

    """
    logger.info(f'Loading raw data from: {file_path}')
    data = pd.read_csv(file_path, low_memory=False)
    logger.info(f'Original shape: {data.shape}')
    return data


def clean_data(data):
    """
    Clean and prepare data

    Parameters
    ----------
    data: Raw pandas DataFrame

    Returns
    -------
    Cleaned pandas DataFrame

    """
    logger.info('Cleaning data...')

    # Filter rows
    values_to_remove = {
        'action_taken': [2, 4, 5, 6, 7, 8],
        'loan_type': [3, 4],
        'applicant_race_1': [1, 2, 4, 6, 7],
        'lien_status': [3, 4],
        'applicant_sex': [3, 4],
        'co_applicant_sex': [3, 4],
        'co_applicant_race_1': [1, 2, 4, 6, 7],
        'applicant_ethnicity': [3, 4],
        'co_applicant_ethnicity': [3, 4]
    }

    data_lending_short = data.copy()
    initial_rows = len(data_lending_short)

    for col, values in values_to_remove.items():
        if col in data_lending_short.columns:
            data_lending_short = data_lending_short[~data_lending_short[col].isin(values)]
        else:
            logger.warning("Column '%s' not found when filtering values", col)

    logger.info(
        'Rows after filtering: %d (removed %d)',
        len(data_lending_short),
        initial_rows - len(data_lending_short),
    )

    # Remove columns
    columns_to_remove = [
        'as_of_year', 'agency_name', 'agency_abbr', 'agency_code', 'property_type_name',
        'property_type', 'owner_occupancy_name', 'owner_occupancy', 'preapproval_name', 'preapproval',
        'state_name', 'state_abbr', 'state_code', 'applicant_race_name_2', 'applicant_race_2',
        'applicant_race_name_3', 'applicant_race_3', 'applicant_race_name_4', 'applicant_race_4',
        'applicant_race_name_5', 'applicant_race_5', 'co_applicant_race_name_2', 'co_applicant_race_2',
        'co_applicant_race_name_3', 'co_applicant_race_3', 'co_applicant_race_name_4', 'co_applicant_race_4',
        'co_applicant_race_name_5', 'co_applicant_race_5', 'purchaser_type_name', 'purchaser_type',
        'denial_reason_name_1', 'denial_reason_1', 'denial_reason_name_2', 'denial_reason_2',
        'denial_reason_name_3', 'denial_reason_3', 'rate_spread', 'hoepa_status_name', 'hoepa_status',
        'edit_status_name', 'edit_status', 'sequence_number', 'application_date_indicator',
        'respondent_id', 'loan_type_name', 'loan_purpose_name', 'action_taken_name', 'msamd_name',
        'county_name', 'applicant_ethnicity_name', 'co_applicant_ethnicity_name',
        'applicant_race_name_1', 'co_applicant_race_name_1', 'applicant_sex_name',
        'co_applicant_sex_name', 'lien_status_name'
    ]

    columns_to_drop = [col for col in columns_to_remove if col in data_lending_short.columns]
    data_lending_dropped = data_lending_short.drop(columns=columns_to_drop)

    logger.info('Columns after dropping: %d', len(data_lending_dropped.columns))

    # Missing values
    rows_before_na = len(data_lending_dropped)
    data_na = data_lending_dropped.dropna(axis='index')

    logger.info(
        'Rows after removing NaN: %d (removed %d)',
        len(data_na),
        rows_before_na - len(data_na),
    )

    data_clean = data_na.copy()

    # Remap action_taken to binary (0 and 1)
    if 'action_taken' in data_na.columns:
        data_clean.loc[:, 'action_taken'] = data_clean['action_taken'].map({1: 0, 3: 1})
        logger.info("Remapped 'action_taken': 1→0 (denied), 3→1 (approved)")
    else:
        logger.warning("'action_taken' column not found during remapping")

    logger.info('Final shape after cleaning: %s', data_clean.shape)
    return data_clean


def create_subsample(data, target_column, sample_size=None, sample_fraction=0.1, random_state=42):
    """
    Create a stratified subsample maintaining class balance

    Parameters
    ----------
    data: pandas DataFrame
    target_column: Name of target variable
    sample_size: Absolute number of samples (optional)
    sample_fraction: Fraction of data to sample (default 0.1 = 10%)
    random_state: Random seed for reproducibility

    Returns
    -------
    Stratified subsample DataFrame

    """
    logger.info('Creating Subsample...')

    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data.")

    # Check class distribution before
    logger.info('Original data shape: %s', data.shape)
    logger.info('Original class distribution:\n%s', data[target_column].value_counts())
    logger.info('Original class proportions:\n%s', data[target_column].value_counts(normalize=True))

    # Calculate sample size
    if sample_size is None:
        sample_size = int(len(data) * sample_fraction)

    # Stratified sampling
    _, sample_data = train_test_split(
        data,
        test_size=sample_size,
        random_state=random_state,
        stratify=data[target_column]
    )

    # Check class distribution after
    logger.info('Sample data shape: %s', sample_data.shape)
    logger.info('Sample class distribution:\n%s', sample_data[target_column].value_counts())
    logger.info('Sample class proportions:\n%s', sample_data[target_column].value_counts(normalize=True))

    return sample_data


def save_clean_data(data, save_path, dataset_name):
    """

    Parameters
    ----------
    data: Cleaned pandas DataFrame
    save_path: Directory to save the file
    dataset_name: Name identifier for the dataset

    Returns
    -------
    File path

    """
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, f'data_clean_{dataset_name}.csv')
    data.to_csv(file_path, index=False)
    logger.info('Cleaned data saved to: %s', file_path)
    return file_path


def preprocess(raw_data_path, save_dir, dataset_name, create_sample=False, sample_fraction=0.1):
    """
    Preprocessing pipeline

    Parameters
    ----------
    raw_data_path: Path to raw data CSV
    save_dir: Directory to save cleaned data
    dataset_name: Name identifier for the dataset
    create_sample: Whether to create a subsample for testing
    sample_fraction: Fraction of data to sample (default 0.1)

    Returns
    -------
    Path to cleaned data file

    """
    logger.info('Data Preprocessing...')

    # Load raw data
    data = load_raw_data(raw_data_path)

    # Clean data
    data_clean = clean_data(data)

    # Create subsample if requested
    if create_sample:
        data_clean = create_subsample(
            data_clean,
            target_column=DATASET_CONFIG['target_column'],
            sample_fraction=sample_fraction,
            random_state=42
        )
        dataset_name = f'{dataset_name}_sample'

    # Save cleaned data
    clean_data_path = save_clean_data(data_clean, save_dir, dataset_name)

    logger.info('Preprocessing Complete')

    return clean_data_path


if __name__ == '__main__':
    raw_path = os.path.join(PATHS['raw_data_dir'], DATASET_CONFIG['raw_data_file'])

    clean_path = preprocess(
        raw_data_path=raw_path,
        save_dir=PATHS['clean_data_dir'],
        dataset_name=DATASET_CONFIG['dataset_name'],
        create_sample = DATASET_CONFIG.get('create_sample', False),
        sample_fraction = DATASET_CONFIG.get('sample_fraction', 0.1)
    )

    logger.info('Cleaned data available at: %s', clean_path)