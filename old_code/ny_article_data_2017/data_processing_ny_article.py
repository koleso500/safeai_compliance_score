import os
import pandas as pd

# Load data and check some general information
data_lending_ny = pd.read_csv(r"C:\Users\koles\Desktop\Master Thesis R project\data_lending.csv", low_memory=False)
print(data_lending_ny.shape)
print(data_lending_ny.columns)
print(data_lending_ny['response'].value_counts())

# Delete and refactor columns
columns_to_remove = ['Unnamed: 0', 'X', 'respondent_id', 'action_taken_name', 'applicant_race_2',
             'applicant_race_3', 'applicant_race_4', 'applicant_race_name_5',
             'applicant_race_5', 'co_applicant_race_2', 'co_applicant_race_3', 'co_applicant_race_name_4',
             'co_applicant_race_4', 'co_applicant_race_name_5', 'co_applicant_race_5', 'denial_reason_name_1',
             'denial_reason_1', 'denial_reason_name_2', 'denial_reason_2', 'denial_reason_name_3',
             'denial_reason_3', 'rate_spread', 'applicant_race_name_2', 'applicant_race_name_3',
             'applicant_race_name_4', 'co_applicant_race_name_2', 'co_applicant_race_name_3','loan_type_name',
             'loan_purpose_name', 'action_taken', 'msamd_name','county_name','applicant_ethnicity_name',
             'co_applicant_ethnicity_name','applicant_race_name_1', 'co_applicant_race_name_1', 'lien_status_name',
             'purchaser_type_name', 'purchaser_type']

data_lending_ny_short = data_lending_ny.drop(columns=columns_to_remove)
data_lending_ny_short['applicant_sex_name'] = data_lending_ny_short['applicant_sex_name'].map({'Male': 1, 'Female': 0})

exclude_vals_1 = ['Information not provided by applicant in mail, Internet, or telephone application', 'Not applicable']
data_lending_ny_short = data_lending_ny_short[~data_lending_ny_short['co_applicant_sex_name'].isin(exclude_vals_1)]
exclude_vals_2 = [1,2,4,6]
data_lending_ny_short = data_lending_ny_short[~data_lending_ny_short['co_applicant_race_1'].isin(exclude_vals_2)]
exclude_vals_3 = [3,4]
data_lending_ny_short = data_lending_ny_short[~data_lending_ny_short['applicant_ethnicity'].isin(exclude_vals_3)]
data_lending_ny_short = data_lending_ny_short[~data_lending_ny_short['co_applicant_ethnicity'].isin(exclude_vals_3)]


data_lending_ny_short['co_applicant_sex_name'] = data_lending_ny_short['co_applicant_sex_name'].map({'Male': 1, 'Female': 0, 'No co-applicant': 3})
print(data_lending_ny_short.columns)

types = data_lending_ny_short.dtypes
print(types)
print(data_lending_ny_short.isna().sum())  # Number of NaNs per column
print(data_lending_ny_short.isna().sum().sum())  # Total number of NaNs in the entire data

# Clean and save data
data_lending_ny_clean = data_lending_ny_short.dropna(axis='index')
print(data_lending_ny_clean.shape)
print(data_lending_ny_clean['response'].value_counts())
for column in data_lending_ny_clean.columns:
    print(f"Value counts for column '{column}':")
    print(data_lending_ny_clean[column].value_counts())
    print("\n" + "-"*40 + "\n")
data_lending_ny_clean.to_csv(os.path.join("../saved_data", "data_lending_clean_ny_article.csv"), index=False)

# Check correlations
correlation_matrix = data_lending_ny_clean.corr()
print(correlation_matrix['response'])

# Create folder for saved data and plots
save_dir = "../saved_data"
os.makedirs(save_dir, exist_ok=True)
os.makedirs("plots", exist_ok=True)