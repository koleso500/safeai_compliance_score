import pandas as pd
import os

from src.config import DATASET_CONFIG, PATHS


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

    print(f"Loading data from: {file_path}")
    data = pd.read_csv(file_path, low_memory=False)
    print(f"Original shape: {data.shape}")
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

    print("\nCleaning data...")

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
            print(f"Warning: Column '{col}' is not found in data")

    print(f"Rows after filtering: {len(data_lending_short)} (removed {initial_rows - len(data_lending_short)})")

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

    print(f"Columns after dropping: {len(data_lending_dropped.columns)}")

    # Missing values
    rows_before_na = len(data_lending_dropped)
    data_na = data_lending_dropped.dropna(axis='index')
    print(f"Rows after removing NaN: {len(data_na)} "
          f"(removed {rows_before_na - len(data_na)})")

    data_clean = data_na.copy()
    # Remap action_taken to binary (0 and 1)
    if 'action_taken' in data_na.columns:
        data_clean.loc[:, 'action_taken'] = data_clean['action_taken'].map({1: 0, 3: 1})
        print("Remapped action_taken: 1→0 (denied), 3→1 (approved)")
    else:
        print("Warning: 'action_taken' column is not found")

    print(f"\nFinal shape after cleaning: {data_clean.shape}")
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
    from sklearn.model_selection import train_test_split

    print("\n" + "=" * 60)
    print("CREATING STRATIFIED SUBSAMPLE")
    print("=" * 60)

    # Check class distribution before
    print(f"\nOriginal data shape: {data.shape}")
    print(f"Original class distribution:")
    print(data[target_column].value_counts())
    print(f"Original class proportions:")
    print(data[target_column].value_counts(normalize=True))

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
    print(f"\nSample data shape: {sample_data.shape}")
    print(f"Sample class distribution:")
    print(sample_data[target_column].value_counts())
    print(f"Sample class proportions:")
    print(sample_data[target_column].value_counts(normalize=True))

    print("=" * 60)

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
    file_path = os.path.join(save_path, f"data_clean_{dataset_name}.csv")
    data.to_csv(file_path, index=False)
    print(f"\nCleaned data saved to: {file_path}")
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

    print("\n" + "=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)

    # Load raw data
    data = load_raw_data(raw_data_path)

    # Clean data
    data_clean = clean_data(data)

    # Create subsample if requested
    if create_sample:
        data_clean = create_subsample(
            data_clean,
            target_column=DATASET_CONFIG["target_column"],
            sample_fraction=sample_fraction,
            random_state=42
        )
        dataset_name = f"{dataset_name}_sample"

    # Save cleaned data
    clean_data_path = save_clean_data(data_clean, save_dir, dataset_name)

    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    return clean_data_path


if __name__ == "__main__":
    raw_path = os.path.join(PATHS["raw_data_dir"], DATASET_CONFIG["raw_data_file"])

    clean_path = preprocess(
        raw_data_path=raw_path,
        save_dir=PATHS["clean_data_dir"],
        dataset_name=DATASET_CONFIG["dataset_name"]
    )

    print(f"\nCleaned data: {clean_path}")