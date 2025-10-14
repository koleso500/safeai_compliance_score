import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import VotingRegressor, StackingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from safeai_files.regression import r_safeai_values
from safeai_files.utils import plot_mean_histogram, plot_model_curves, plot_diff_mean_histogram

# Data loading and basic information
data = pd.read_excel("employee.xlsx")
print(data.shape)
print(data.columns)
print("This dataset has {} rows and {} columns".format(data.shape[0], data.shape[1]))
types = data.dtypes
print(types)

# Check NaNs
print(data.isna().sum())  # Number of NaNs per column
print(data.isna().sum().sum()) # Total number of NaNs in the entire data

# Change to int
print(data.head())
print(data["gender"].value_counts())
print(data["minority"].value_counts())

data["gender"] = np.where(data["gender"]=="m", 0, 1)
data["minority"] = np.where(data["minority"]=="no_min", 0, 1)

# Column for Regression task
data["salary_growth"] = data["salary"]-data["startsal"]
data.drop(["salary", "startsal"], axis=1, inplace=True)
print(data.head())

# Split to train and test
x = data.drop("salary_growth", axis=1)
y_reg = data["salary_growth"]

x_train, x_test, y_train_rg, y_test_rg = train_test_split(x, y_reg, test_size=0.3, random_state=1)

# Regression problem
# Linear Regression
reg_model = LinearRegression()
reg_model.fit(x_train, y_train_rg)
y_prob_reg = reg_model.predict(x_test)
results_reg = r_safeai_values(x_train, x_test, y_test_rg, y_prob_reg, reg_model, "Employee Regression", "plots")
print(results_reg)

# Random Forest
rf_rg = RandomForestRegressor(random_state=123)
rf_rg.fit(x_train, y_train_rg)
y_prob_rf_rg = rf_rg.predict(x_test)
results_rf_rg = r_safeai_values(x_train, x_test, y_test_rg, y_prob_rf_rg, rf_rg, "Employee Regression", "plots")
print(results_rf_rg)

# XGBoosting
xgb_rg = xgb.XGBRegressor(random_state=123)
xgb_rg.fit(x_train, y_train_rg)
y_prob_xgb_rg = xgb_rg.predict(x_test)
results_xgb_rg = r_safeai_values(x_train, x_test, y_test_rg, y_prob_xgb_rg, xgb_rg, "Employee Regression", "plots")
print(results_xgb_rg)

# Stacked Ensemble Model
stacking_rg = StackingRegressor(estimators=[('rf', rf_rg), ('xgb', xgb_rg)], final_estimator=reg_model)
stacking_rg.fit(x_train, y_train_rg)
y_prob_se_rg = stacking_rg.predict(x_test)
results_se_rg = r_safeai_values(x_train, x_test, y_test_rg, y_prob_se_rg, stacking_rg, "Employee Regression", "plots")
print(results_se_rg)

# Voting Ensemble Model
voting_rg = VotingRegressor(estimators=[('rf', rf_rg), ('xgb', xgb_rg)])
voting_rg.fit(x_train, y_train_rg)
y_prob_ve_rg = voting_rg.predict(x_test)
results_ve_rg = r_safeai_values(x_train, x_test, y_test_rg, y_prob_ve_rg, voting_rg, "Employee Regression", "plots")
print(results_ve_rg)

# Random Model
random_rg = DummyRegressor()
random_rg.fit(x_train, y_train_rg)
y_prob_r_rg = random_rg.predict(x_test)
results_r_rg = r_safeai_values(x_train, x_test, y_test_rg, y_prob_r_rg, random_rg, "Employee Regression", "plots")
print(results_r_rg)

# Extract values
x_lr = results_reg["x_final"]
y_lr = results_reg["y_final"]
z_lr = results_reg["z_final"]

x_rf = results_rf_rg["x_final"]
y_rf = results_rf_rg["y_final"]
z_rf = results_rf_rg["z_final"]

x_xgb = results_xgb_rg["x_final"]
y_xgb = results_xgb_rg["y_final"]
z_xgb = results_xgb_rg["z_final"]

x_se = results_se_rg["x_final"]
y_se = results_se_rg["y_final"]
z_se = results_se_rg["z_final"]

x_ve = results_ve_rg["x_final"]
y_ve = results_ve_rg["y_final"]
z_ve = results_ve_rg["z_final"]

x_r = results_r_rg["x_final"]
y_r = results_r_rg["y_final"]
z_r = results_r_rg["z_final"]

# Differences
x_lr_r = (np.array(x_lr) - np.array(x_r)).tolist()
y_lr_r = (np.array(y_lr) - np.array(y_r)).tolist()
z_lr_r = (np.array(z_lr) - np.array(z_r)).tolist()

x_rf_r = (np.array(x_rf) - np.array(x_r)).tolist()
y_rf_r = (np.array(y_rf) - np.array(y_r)).tolist()
z_rf_r = (np.array(z_rf) - np.array(z_r)).tolist()

x_xgb_r = (np.array(x_xgb) - np.array(x_r)).tolist()
y_xgb_r = (np.array(y_xgb) - np.array(y_r)).tolist()
z_xgb_r = (np.array(z_xgb) - np.array(z_r)).tolist()

x_se_r = (np.array(x_se) - np.array(x_r)).tolist()
y_se_r = (np.array(y_se) - np.array(y_r)).tolist()
z_se_r = (np.array(z_se) - np.array(z_r)).tolist()

x_ve_r = (np.array(x_ve) - np.array(x_r)).tolist()
y_ve_r = (np.array(y_ve) - np.array(y_r)).tolist()
z_ve_r = (np.array(z_ve) - np.array(z_r)).tolist()

# Compliance Curves (Regression)
# Linear Regression
x_step = np.linspace(0, 1, len(y_r))
plot_model_curves(x_step,[x_lr, y_lr, z_lr], model_name="Linear Regression",
                  title="Linear Regression Curves (Regression)")

# Random Forest
plot_model_curves(x_step,[x_rf, y_rf, z_rf], model_name="Random Forest",
                  title="Random Forest Curves (Regression)")

# XGBoosting
plot_model_curves(x_step,[x_xgb, y_xgb, z_xgb], model_name="XGBoosting",
                  title="XGBoosting Curves (Regression)")

# Stacked Ensemble Model
plot_model_curves(x_step,[x_se, y_se, z_se], model_name="Stacked Ensemble Model",
                  title="Stacked Ensemble Model Curves (Regression)")

# Voting Ensemble Model
plot_model_curves(x_step,[x_ve, y_ve, z_ve], model_name="Voting Ensemble Model",
                  title="Voting Ensemble Model Curves (Regression)")

# Random Model
plot_model_curves(x_step,[x_r, y_r, z_r], model_name="Random Model",
                  title="Random Model Curves (Regression)")

# Difference LR and Random
plot_model_curves(x_step,[x_lr_r, y_lr_r, z_lr_r], model_name="Random", prefix="Difference",
                  title="LR and Random Curves Difference (Regression)")

# Difference RF and Random
plot_model_curves(x_step,[x_rf_r, y_rf_r, z_rf_r], model_name="Random", prefix="Difference",
                  title="RF and Random Curves Difference (Regression)")

# Difference XGB and Random
plot_model_curves(x_step,[x_xgb_r, y_xgb_r, z_xgb_r], model_name="Random", prefix="Difference",
                  title="XGB and Random Curves Difference (Regression)")

# Difference SE and Random
plot_model_curves(x_step,[x_se_r, y_se_r, z_se_r], model_name="Random", prefix="Difference",
                  title="SE and Random Curves Difference (Regression)")

# Difference VE and Random
plot_model_curves(x_step,[x_ve_r, y_ve_r, z_ve_r], model_name="Random", prefix="Difference",
                  title="VE and Random Curves Difference (Regression)")

plt.show()

# Values and Volume
rgas_lr = np.array(x_lr)
rges_lr = np.array(y_lr)
rgrs_lr = np.array(z_lr)

rgas_rf = np.array(x_rf)
rges_rf = np.array(y_rf)
rgrs_rf = np.array(z_rf)

rgas_xgb = np.array(x_xgb)
rges_xgb = np.array(y_xgb)
rgrs_xgb = np.array(z_xgb)

rgas_se = np.array(x_se)
rges_se = np.array(y_se)
rgrs_se = np.array(z_se)

rgas_ve = np.array(x_ve)
rges_ve = np.array(y_ve)
rgrs_ve = np.array(z_ve)

rgas_random = np.array(x_r)
rges_random = np.array(y_r)
rgrs_random = np.array(z_r)

# Scalar fields, matrix of initial values
rga_lr, rge_lr, rgr_lr = np.meshgrid(rgas_lr, rges_lr, rgrs_lr, indexing='ij')
rga_rf, rge_rf, rgr_rf = np.meshgrid(rgas_rf, rges_rf, rgrs_rf, indexing='ij')
rga_xgb, rge_xgb, rgr_xgb = np.meshgrid(rgas_xgb, rges_xgb, rgrs_xgb, indexing='ij')
rga_se, rge_se, rgr_se = np.meshgrid(rgas_se, rges_se, rgrs_se, indexing='ij')
rga_ve, rge_ve, rgr_ve = np.meshgrid(rgas_ve, rges_ve, rgrs_ve, indexing='ij')
rga_r, rge_r, rgr_r = np.meshgrid(rgas_random, rges_random, rgrs_random, indexing='ij')

# Means
models = [
    ((rga_lr,  rge_lr,  rgr_lr),  "Logistic Regression", "Logistic Regression"),
    ((rga_rf,  rge_rf,  rgr_rf),  "Random Forest", "Random Forest Model"),
    ((rga_xgb, rge_xgb, rgr_xgb), "XGBoosting", "XGB Model"),
    ((rga_se,  rge_se,  rgr_se),  "Stacked Ensemble", "Stacked Ensemble Model"),
    ((rga_ve,  rge_ve,  rgr_ve),  "Voting Ensemble", "Voting Ensemble Model"),
    ((rga_r,   rge_r,   rgr_r),   "Random Classifier", "Random Classifier"),
]

# All arithmetic mean histograms
for (rga_var, rge_var, rgr_var), model_name, bar_label in models:
    plot_mean_histogram(
        rga_var, rge_var, rgr_var,
        model_name=model_name,
        bar_label=bar_label,
        mean_type="arithmetic"
    )
plt.show()

# All geometric mean histograms
for (rga_var, rge_var, rgr_var), model_name, bar_label in models:
    plot_mean_histogram(
        rga_var, rge_var, rgr_var,
        model_name=model_name,
        bar_label=bar_label,
        mean_type="geometric"
    )
plt.show()

# All quadratic mean histograms
for (rga_var, rge_var, rgr_var), model_name, bar_label in models:
    plot_mean_histogram(
        rga_var, rge_var, rgr_var,
        model_name=model_name,
        bar_label=bar_label,
        mean_type="quadratic"
    )
plt.show()

# Differences Means
# Values
rga_d_lr = np.array(x_lr_r)
rge_d_lr = np.array(y_lr_r)
rgr_d_lr = np.array(z_lr_r)

rga_d_rf = np.array(x_rf_r)
rge_d_rf = np.array(y_rf_r)
rgr_d_rf = np.array(z_rf_r)

rga_d_xgb = np.array(x_xgb_r)
rge_d_xgb = np.array(y_xgb_r)
rgr_d_xgb = np.array(z_xgb_r)

rga_d_se = np.array(x_se_r)
rge_d_se = np.array(y_se_r)
rgr_d_se = np.array(z_se_r)

rga_d_ve = np.array(x_ve_r)
rge_d_ve = np.array(y_ve_r)
rgr_d_ve = np.array(z_ve_r)

rga_lr_d, rge_lr_d, rgr_lr_d = np.meshgrid(rga_d_lr, rge_d_lr, rgr_d_lr, indexing='ij')
rga_rf_d, rge_rf_d, rgr_rf_d = np.meshgrid(rga_d_rf, rge_d_rf, rgr_d_rf, indexing='ij')
rga_xgb_d, rge_xgb_d, rgr_xgb_d = np.meshgrid(rga_d_xgb, rge_d_xgb, rgr_d_xgb, indexing='ij')
rga_se_d, rge_se_d, rgr_se_d = np.meshgrid(rga_d_se, rge_d_se, rgr_d_se, indexing='ij')
rga_ve_d, rge_ve_d, rgr_ve_d = np.meshgrid(rga_d_ve, rge_d_ve, rgr_d_ve, indexing='ij')

models_diff = [
    ((rga_lr_d,  rge_lr_d,  rgr_lr_d),  "Logistic Regression", "Logistic Regression"),
    ((rga_rf_d,  rge_rf_d,  rgr_rf_d),  "Random Forest", "Random Forest Model"),
    ((rga_xgb_d, rge_xgb_d, rgr_xgb_d), "XGBoosting", "XGB Model"),
    ((rga_se_d,  rge_se_d,  rgr_se_d),  "Stacked Ensemble", "Stacked Ensemble Model"),
    ((rga_ve_d,  rge_ve_d,  rgr_ve_d),  "Voting Ensemble", "Voting Ensemble Model"),
]

# All arithmetic mean differences histograms
for (rga_var, rge_var, rgr_var), model_name, bar_label in models_diff:
    plot_diff_mean_histogram(
        rga_var, rge_var, rgr_var,
        model_name=model_name,
        bar_label=bar_label,
        mean_type="arithmetic"
    )
plt.show()

# All geometric mean differences histograms
for (rga_var, rge_var, rgr_var), model_name, bar_label in models_diff:
    plot_diff_mean_histogram(
        rga_var, rge_var, rgr_var,
        model_name=model_name,
        bar_label=bar_label,
        mean_type="geometric"
    )
plt.show()

# All quadratic mean differences histograms
for (rga_var, rge_var, rgr_var), model_name, bar_label in models_diff:
    plot_diff_mean_histogram(
        rga_var, rge_var, rgr_var,
        model_name=model_name,
        bar_label=bar_label,
        mean_type="quadratic"
    )
plt.show()

# Hypervolume approach
def hypervolume(x1, y2, z3):
    v1 = np.array(x1)
    v2 = np.array(y2)
    v3 = np.array(z3)

    # Construct Gram matrix
    g = np.array([
        [np.dot(v1, v1), np.dot(v1, v2), np.dot(v1, v3)],
        [np.dot(v2, v1), np.dot(v2, v2), np.dot(v2, v3)],
        [np.dot(v3, v1), np.dot(v3, v2), np.dot(v3, v3)]
    ])

    # Hypervolume
    volume = np.sqrt(np.linalg.det(g))

    return volume

# Hypervolume LR
volume_lr = hypervolume(rgas_lr, rges_lr, rgrs_lr)
print(f"Hypervolume LR: {volume_lr:.3f}")

# Hypervolume RF
volume_rf = hypervolume(rgas_rf, rges_rf, rgrs_rf)
print(f"Hypervolume RF: {volume_rf:.3f}")

# Hypervolume XGB
volume_xgb = hypervolume(rgas_xgb, rges_xgb, rgrs_xgb)
print(f"Hypervolume XGB: {volume_xgb:.3f}")

# Hypervolume SE
volume_se = hypervolume(rgas_se, rges_se, rgrs_se)
print(f"Hypervolume SE: {volume_se:.3f}")

# Hypervolume VE
volume_ve = hypervolume(rgas_ve, rges_ve, rgrs_ve)
print(f"Hypervolume VE: {volume_ve:.3f}")

# Hypervolume R
volume_r = hypervolume(rgas_random, rges_random, rgrs_random)
print(f"Hypervolume Random: {volume_r:.3f}")

# TOPSIS approach
best_x_list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
worst_x_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

best_y_list = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
worst_y_list = [1.0, 0.9736842105263158, 0.9473684210526316, 0.9210526315789473, 0.8947368421052632, 0.868421052631579, 0.8421052631578947, 0.8157894736842105, 0.7894736842105263, 0.7631578947368421, 0.736842105263158, 0.7105263157894737, 0.6842105263157895, 0.6578947368421053, 0.631578947368421, 0.6052631578947368, 0.5789473684210527, 0.5526315789473685, 0.5263157894736843, 0.5 ]

best_z_list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
worst_z_list = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

x_plus = np.mean(best_x_list)
x_minus = np.mean(worst_x_list)

y_plus = np.mean(best_y_list)
y_minus = np.mean(worst_y_list)

z_plus = np.mean(best_z_list)
z_minus = np.mean(worst_z_list)

mean_x_lr = np.mean(rgas_lr)
mean_y_lr = np.mean(rges_lr)
mean_z_lr = np.mean(rgrs_lr)

mean_x_rf = np.mean(rgas_rf)
mean_y_rf = np.mean(rges_rf)
mean_z_rf = np.mean(rgrs_rf)

mean_x_xgb = np.mean(rgas_xgb)
mean_y_xgb = np.mean(rges_xgb)
mean_z_xgb = np.mean(rgrs_xgb)

mean_x_se = np.mean(rgas_se)
mean_y_se = np.mean(rges_se)
mean_z_se = np.mean(rgrs_se)

mean_x_ve = np.mean(rgas_ve)
mean_y_ve = np.mean(rges_ve)
mean_z_ve = np.mean(rgrs_ve)

mean_x_r = np.mean(rgas_random)
mean_y_r = np.mean(rges_random)
mean_z_r = np.mean(rgrs_random)

means = {
    "Logistic":       (mean_x_lr, mean_y_lr, mean_z_lr),
    "RandomForest":   (mean_x_rf, mean_y_rf, mean_z_rf),
    "XGBoost":        (mean_x_xgb, mean_y_xgb, mean_z_xgb),
    "StackedEnsemble":(mean_x_se, mean_y_se, mean_z_se),
    "VotingEnsemble": (mean_x_ve,  mean_y_ve,  mean_z_ve),
    "RandomBaseline": (mean_x_r,  mean_y_r,  mean_z_r),
}

df = pd.DataFrame.from_dict(
    means,
    orient="index",
    columns=["mean_x", "mean_y", "mean_z"]
)

for col in ["mean_x", "mean_y", "mean_z"]:
    vec = df[col].values.astype(float)
    norm = np.sqrt((vec**2).sum())
    df["r_" + col] = vec / norm

weights = np.array([1/3, 1/3, 1/3])
df["v_mean_x"] = df["r_mean_x"] * weights[0]
df["v_mean_y"] = df["r_mean_y"] * weights[1]
df["v_mean_z"] = df["r_mean_z"] * weights[2]

norm_x = np.sqrt((df["mean_x"].values ** 2).sum())
norm_y = np.sqrt((df["mean_y"].values ** 2).sum())
norm_z = np.sqrt((df["mean_z"].values ** 2).sum())

r_x_plus = x_plus / norm_x
r_x_minus = x_minus / norm_x

r_y_plus = y_plus / norm_y
r_y_minus = y_minus / norm_y

r_z_plus = z_plus / norm_z
r_z_minus = z_minus / norm_z

v_x_plus  = r_x_plus  * weights[0]
v_x_minus = r_x_minus * weights[0]

v_y_plus  = r_y_plus  * weights[1]
v_y_minus = r_y_minus * weights[1]

v_z_plus  = r_z_plus  * weights[2]
v_z_minus = r_z_minus * weights[2]

df["S_plus"] = np.sqrt(
    (df["v_mean_x"] - v_x_plus)**2 +
    (df["v_mean_y"] - v_y_plus)**2 +
    (df["v_mean_z"] - v_z_plus)**2
)

df["S_minus"] = np.sqrt(
    (df["v_mean_x"] - v_x_minus)**2 +
    (df["v_mean_y"] - v_y_minus)**2 +
    (df["v_mean_z"] - v_z_minus)**2
)

df["C"] = df["S_minus"] / (df["S_plus"] + df["S_minus"])
df["Rank"] = df["C"].rank(ascending=False)
df_sorted = df.sort_values("C", ascending=False)
print(df_sorted[["C", "Rank"]])