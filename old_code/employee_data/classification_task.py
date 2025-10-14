import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib.gridspec import GridSpec
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from safeai_files.check_compliance import safeai_values
from safeai_files.utils import plot_mean_histogram, plot_model_curves, plot_diff_mean_histogram, save_model_metrics

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

# Column for Classification task
data["doubling_salary"] = np.where(data["salary"]/data["startsal"] > 2,1,0)
print(data["doubling_salary"].value_counts())
data.drop(["salary", "startsal"], axis=1, inplace=True)
print(data.head())

# Split to train and test
x = data.drop("doubling_salary", axis=1)
y_class = data["doubling_salary"]

x_train, x_test, y_train_cl, y_test_cl = train_test_split(x, y_class, test_size=0.3, random_state=1)

# Classification problem
# Logistic Regression
log_model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=123)
log_model.fit(x_train, y_train_cl)
y_prob_lr = log_model.predict_proba(x_test)[:, 1]
results_log = safeai_values(x_train, x_test, y_test_cl, y_prob_lr, log_model, "Employee Classification", "plots")
print(results_log)
save_model_metrics(results_log)

# Random Forest
rf_model = RandomForestClassifier(random_state=123)
rf_model.fit(x_train, y_train_cl)
y_prob_rf = rf_model.predict_proba(x_test)[:, 1]
results_rf = safeai_values(x_train, x_test, y_test_cl, y_prob_rf, rf_model, "Employee Classification", "plots")
print(results_rf)
save_model_metrics(results_rf)

# XGBoosting
xgb_model = xgb.XGBClassifier(random_state=123)
xgb_model.fit(x_train, y_train_cl)
y_prob_xgb = xgb_model.predict_proba(x_test)[:, 1]
results_xgb = safeai_values(x_train, x_test, y_test_cl, y_prob_xgb, xgb_model, "Employee Classification", "plots")
print(results_xgb)
save_model_metrics(results_xgb)

# Stacked Ensemble Model
stacking_clf = StackingClassifier(estimators=[('rf', rf_model), ('xgb', xgb_model)], final_estimator=log_model)
stacking_clf.fit(x_train, y_train_cl)
y_prob_se = stacking_clf.predict_proba(x_test)[:, 1]
results_se = safeai_values(x_train, x_test, y_test_cl, y_prob_se, stacking_clf, "Employee Classification", "plots")
print(results_se)
save_model_metrics(results_se)

# Voting Ensemble Model
voting_clf = VotingClassifier(estimators=[('rf', rf_model), ('xgb', xgb_model)], voting='soft')
voting_clf.fit(x_train, y_train_cl)
y_prob_ve = voting_clf.predict_proba(x_test)[:, 1]
results_ve = safeai_values(x_train, x_test, y_test_cl, y_prob_ve, voting_clf, "Employee Classification", "plots")
print(results_ve)
save_model_metrics(results_ve)

# Random Model
random_model = DummyClassifier(random_state=123)
random_model.fit(x_train, y_train_cl)
y_prob_r = random_model.predict_proba(x_test)[:, 1]
results_r = safeai_values(x_train, x_test, y_test_cl, y_prob_r, random_model, "Employee Classification", "plots")
print(results_r)
save_model_metrics(results_r)

# Extract values
x_lr = results_log["x_final"]
y_lr = results_log["y_final"]
z_lr = results_log["z_final"]

x_rf = results_rf["x_final"]
y_rf = results_rf["y_final"]
z_rf = results_rf["z_final"]

x_xgb = results_xgb["x_final"]
y_xgb = results_xgb["y_final"]
z_xgb = results_xgb["z_final"]

x_se = results_se["x_final"]
y_se = results_se["y_final"]
z_se = results_se["z_final"]

x_ve = results_ve["x_final"]
y_ve = results_ve["y_final"]
z_ve = results_ve["z_final"]

x_r = results_r["x_final"]
y_r = results_r["y_final"]
z_r = results_r["z_final"]

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

# Compliance Curves (Classification)
# Logistic Regression
fig, axs = plt.subplots(3, 2, figsize=(15, 20))
axs = axs.flatten()

x_step = np.linspace(0, 1, len(y_r))
plot_model_curves(x_step, [x_lr, y_lr, z_lr], model_name="LR",
                  title="Logistic Regression Curves (Employee)", ax=axs[0])

# Random Forest
plot_model_curves(x_step, [x_rf, y_rf, z_rf], model_name="RF",
                  title="Random Forest Curves (Employee)", ax=axs[1])

# XGBoosting
plot_model_curves(x_step,[x_xgb, y_xgb, z_xgb], model_name="XGB",
                  title="XGBoosting Curves (Employee)", ax=axs[2])

# Stacked Ensemble Model
plot_model_curves(x_step,[x_se, y_se, z_se], model_name="SE",
                  title="Stacked Ensemble Model Curves (Employee)", ax=axs[3])

# Voting Ensemble Model
plot_model_curves(x_step,[x_ve, y_ve, z_ve], model_name="VE",
                  title="Voting Ensemble Model Curves (Employee)", ax=axs[4])

# Random Model
plot_model_curves(x_step,[x_r, y_r, z_r], model_name="Random",
                  title="Random Model Curves (Employee)", ax=axs[5])

plt.tight_layout()
fig.subplots_adjust(top=0.90, hspace=0.4)
plt.show()


# Difference
# Create figure with custom grid layout
fig = plt.figure(figsize=(20, 12))
gs = GridSpec(2, 6, figure=fig, hspace=0.4, wspace=0.4)

# First row: 3 plots (equal width)
ax1 = fig.add_subplot(gs[0, 0:2])  # Columns 0-1
ax2 = fig.add_subplot(gs[0, 2:4])  # Columns 2-3
ax3 = fig.add_subplot(gs[0, 4:6])  # Columns 4-5

# Second row: 2 plots (centered)
ax4 = fig.add_subplot(gs[1, 1:3])  # Columns 1-2 (centered)
ax5 = fig.add_subplot(gs[1, 3:5])  # Columns 3-4 (centered)

# Your plotting calls
# First row (3 plots)
plot_model_curves(x_step, [x_lr_r, y_lr_r, z_lr_r], model_name="Random", prefix="Difference",
                  title="LR and Random Curves Difference (Employee)", ax=ax1)

plot_model_curves(x_step, [x_rf_r, y_rf_r, z_rf_r], model_name="Random", prefix="Difference",
                  title="RF and Random Curves Difference (Employee)", ax=ax2)

plot_model_curves(x_step, [x_xgb_r, y_xgb_r, z_xgb_r], model_name="Random", prefix="Difference",
                  title="XGB and Random Curves Difference (Employee)", ax=ax3)

# Second row (2 plots, centered)
plot_model_curves(x_step, [x_se_r, y_se_r, z_se_r], model_name="Random", prefix="Difference",
                  title="SE and Random Curves Difference (Employee)", ax=ax4)

plot_model_curves(x_step, [x_ve_r, y_ve_r, z_ve_r], model_name="Random", prefix="Difference",
                  title="VE and Random Curves Difference (Employee)", ax=ax5)

fig.subplots_adjust(top=0.90)
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
fig, axs = plt.subplots(3, 2, figsize=(15, 20))
axs = axs.flatten()
for i, ((rga_var, rge_var, rgr_var), model_name, bar_label) in enumerate(models):
    plot_mean_histogram(
        rga_var, rge_var, rgr_var,
        model_name=model_name,
        bar_label=bar_label,
        mean_type="arithmetic",
        ax=axs[i]
    )
plt.tight_layout()
fig.subplots_adjust(top=0.90, hspace=0.4)
plt.show()

# All geometric mean histograms
fig, axs = plt.subplots(3, 2, figsize=(15, 20))
axs = axs.flatten()
for i, ((rga_var, rge_var, rgr_var), model_name, bar_label) in enumerate(models):
    plot_mean_histogram(
        rga_var, rge_var, rgr_var,
        model_name=model_name,
        bar_label=bar_label,
        mean_type="geometric",
        ax=axs[i]
    )
plt.tight_layout()
fig.subplots_adjust(top=0.90, hspace=0.4)
plt.show()

# All quadratic mean histograms
fig, axs = plt.subplots(3, 2, figsize=(15, 20))
axs = axs.flatten()
for i, ((rga_var, rge_var, rgr_var), model_name, bar_label) in enumerate(models):
    plot_mean_histogram(
        rga_var, rge_var, rgr_var,
        model_name=model_name,
        bar_label=bar_label,
        mean_type="quadratic",
        ax=axs[i]
    )
plt.tight_layout()
fig.subplots_adjust(top=0.90, hspace=0.4)
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
    ((rga_lr_d,  rge_lr_d,  rgr_lr_d),  "LR", "Logistic Regression"),
    ((rga_rf_d,  rge_rf_d,  rgr_rf_d),  "RF", "Random Forest Model"),
    ((rga_xgb_d, rge_xgb_d, rgr_xgb_d), "XGB", "XGB Model"),
    ((rga_se_d,  rge_se_d,  rgr_se_d),  "SE", "Stacked Ensemble Model"),
    ((rga_ve_d,  rge_ve_d,  rgr_ve_d),  "VE", "Voting Ensemble Model"),
]

# All arithmetic mean differences histograms
fig = plt.figure(figsize=(20, 20))
gs = GridSpec(2, 6, figure=fig, hspace=0.4, wspace=2)

# First row: 3 plots (equal width)
ax1 = fig.add_subplot(gs[0, 0:2])  # Columns 0-1
ax2 = fig.add_subplot(gs[0, 2:4])  # Columns 2-3
ax3 = fig.add_subplot(gs[0, 4:6])  # Columns 4-5

# Second row: 2 plots (centered)
ax4 = fig.add_subplot(gs[1, 1:3])  # Columns 1-2 (centered)
ax5 = fig.add_subplot(gs[1, 3:5])  # Columns 3-4 (centered)

# Create axes list for easy iteration
axs = [ax1, ax2, ax3, ax4, ax5]

for i, ((rga_var, rge_var, rgr_var), model_name, bar_label) in enumerate(models_diff):
    plot_diff_mean_histogram(
        rga_var, rge_var, rgr_var,
        model_name=model_name,
        bar_label=bar_label,
        mean_type="arithmetic",
        ax=axs[i]
    )

plt.tight_layout()
fig.subplots_adjust(top=0.90, hspace=0.4)
plt.show()

# All geometric mean differences histograms
fig = plt.figure(figsize=(20, 20))
gs = GridSpec(2, 6, figure=fig, hspace=0.4, wspace=2)

# First row: 3 plots (equal width)
ax1 = fig.add_subplot(gs[0, 0:2])  # Columns 0-1
ax2 = fig.add_subplot(gs[0, 2:4])  # Columns 2-3
ax3 = fig.add_subplot(gs[0, 4:6])  # Columns 4-5

# Second row: 2 plots (centered)
ax4 = fig.add_subplot(gs[1, 1:3])  # Columns 1-2 (centered)
ax5 = fig.add_subplot(gs[1, 3:5])  # Columns 3-4 (centered)

# Create axes list for easy iteration
axs = [ax1, ax2, ax3, ax4, ax5]

for i, ((rga_var, rge_var, rgr_var), model_name, bar_label) in enumerate(models_diff):
    plot_diff_mean_histogram(
        rga_var, rge_var, rgr_var,
        model_name=model_name,
        bar_label=bar_label,
        mean_type="geometric",
        ax=axs[i]
    )

plt.tight_layout()
fig.subplots_adjust(top=0.90, hspace=0.4)
plt.show()

# All quadratic mean differences histograms
fig = plt.figure(figsize=(20, 20))
gs = GridSpec(2, 6, figure=fig, hspace=0.4, wspace=2)

# First row: 3 plots (equal width)
ax1 = fig.add_subplot(gs[0, 0:2])  # Columns 0-1
ax2 = fig.add_subplot(gs[0, 2:4])  # Columns 2-3
ax3 = fig.add_subplot(gs[0, 4:6])  # Columns 4-5

# Second row: 2 plots (centered)
ax4 = fig.add_subplot(gs[1, 1:3])  # Columns 1-2 (centered)
ax5 = fig.add_subplot(gs[1, 3:5])  # Columns 3-4 (centered)

# Create axes list for easy iteration
axs = [ax1, ax2, ax3, ax4, ax5]

for i, ((rga_var, rge_var, rgr_var), model_name, bar_label) in enumerate(models_diff):
    plot_diff_mean_histogram(
        rga_var, rge_var, rgr_var,
        model_name=model_name,
        bar_label=bar_label,
        mean_type="quadratic",
        ax=axs[i]
    )

plt.tight_layout()
fig.subplots_adjust(top=0.90, hspace=0.4, wspace=4)
plt.show()

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