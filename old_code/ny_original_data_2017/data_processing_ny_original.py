import os
import pandas as pd

# Load data and check some general information
data_lending_ny = pd.read_csv(r"C:\Users\koles\Desktop\Master Thesis R project\hmda_2017_ny_all-records_labels.csv", low_memory=False)
print(data_lending_ny.shape)
print(data_lending_ny.columns)
print(data_lending_ny['action_taken_name'].value_counts())

# Values to remove
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

# Filter
data_lending_ny_short = data_lending_ny.copy()
for col, values in values_to_remove.items():
    data_lending_ny_short = data_lending_ny_short[~data_lending_ny_short[col].isin(values)]
print(data_lending_ny_short.shape)

# Delete and refactor columns
columns_to_remove = ['as_of_year', 'agency_name', 'agency_abbr', 'agency_code', 'property_type_name',
       'property_type', 'owner_occupancy_name', 'owner_occupancy', 'preapproval_name', 'preapproval',
       'state_name', 'state_abbr', 'state_code', 'applicant_race_name_2', 'applicant_race_2',
       'applicant_race_name_3', 'applicant_race_3', 'applicant_race_name_4', 'applicant_race_4',
       'applicant_race_name_5', 'applicant_race_5', 'co_applicant_race_name_2', 'co_applicant_race_2',
       'co_applicant_race_name_3', 'co_applicant_race_3', 'co_applicant_race_name_4', 'co_applicant_race_4',
       'co_applicant_race_name_5', 'co_applicant_race_5', 'purchaser_type_name', 'purchaser_type',
       'denial_reason_name_1', 'denial_reason_1', 'denial_reason_name_2', 'denial_reason_2', 'denial_reason_name_3',
       'denial_reason_3', 'rate_spread', 'hoepa_status_name', 'hoepa_status','edit_status_name', 'edit_status',
       'sequence_number', 'application_date_indicator', 'respondent_id', 'loan_type_name', 'loan_purpose_name',
       'action_taken_name', 'msamd_name', 'county_name', 'applicant_ethnicity_name', 'co_applicant_ethnicity_name',
       'applicant_race_name_1', 'co_applicant_race_name_1', 'applicant_sex_name', 'co_applicant_sex_name',
       'lien_status_name']

data_lending_ny_dropped = data_lending_ny_short.drop(columns=columns_to_remove)
print(data_lending_ny_dropped.shape)
print(data_lending_ny_dropped.columns)

types = data_lending_ny_dropped.dtypes
print(types)
print(data_lending_ny_dropped.isna().sum())  # Number of NaNs per column
print(data_lending_ny_dropped.isna().sum().sum())  # Total number of NaNs in the entire data

# Clean data
data_lending_ny_clean = data_lending_ny_dropped.dropna(axis='index')
data_lending_ny_clean.loc[:, 'action_taken'] = data_lending_ny_clean['action_taken'].map({1: 0, 3: 1})
print(data_lending_ny_clean.shape)
print(data_lending_ny_clean['action_taken'].value_counts())
for column in data_lending_ny_clean.columns:
    print(f"Value counts for column '{column}':")
    print(data_lending_ny_clean[column].value_counts())
    print("\n" + "-"*40 + "\n")
data_lending_ny_clean.to_csv(os.path.join("../saved_data", "data_lending_clean_ny_original.csv"), index=False)

# Check correlations
correlation_matrix = data_lending_ny_clean.corr()
print(correlation_matrix['action_taken'])

# Create folder for saved data and plots
save_dir = "../saved_data"
os.makedirs(save_dir, exist_ok=True)
os.makedirs("plots", exist_ok=True)
