"""
Customer Churn Prediction System — Consolidated Analytics & ML Pipeline
=======================================================================
This file brings together all code snippets from all four project phases
into a single, runnable Python script. Tkinter has been removed as the UI
is now served through a modern web interface (index.html + server.py).

Phases covered:
  Phase 1 — Project setup, dataset loading, and initial inspection
  Phase 2 — Data cleaning, EDA, and visualization (6 charts saved as PNG)
  Phase 3 — Classification (LR, DT, RF) + Regression + supplementary snippet
  Phase 4 — Final model selection, serialization, and deployment prep
"""

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORTS — libraries used across all four phases
# ─────────────────────────────────────────────────────────────────────────────
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no display needed
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)


# ─────────────────────────────────────────────────────────────────────────────
#  PATH SETUP — locate CSV files relative to this script
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV  = os.path.join(BASE_DIR, "customer_churn_dataset-training-master.csv")
TEST_CSV   = os.path.join(BASE_DIR, "customer_churn_dataset-testing-master.csv")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
PLOT_DIR   = os.path.join(BASE_DIR, "eda_plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,   exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 1 — Dataset loading and first look at the data
#  (The project proposal defines the dataset as sourced from Kaggle;
#   this is the very first time we read the CSV and inspect the raw rows.)
# ─────────────────────────────────────────────────────────────────────────────

# Phase 1 — load the training dataset and print all its rows for initial inspection
data = pd.read_csv(TRAIN_CSV)
print("=" * 60)
print("PHASE 1 — First look at the dataset")
print("=" * 60)
print(data)


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 2 — Data Preparation & Exploratory Data Analysis
#  (Data cleaning, descriptive stats, univariate/bivariate/correlation analysis
#   — mirrors the 9 Jupyter Notebook cells from the Phase 2 report.)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("PHASE 2 — Data Preparation & EDA")
print("=" * 60)

# --- Phase 2, Cell 1 ---
# Reload the dataset fresh — this is the canonical Phase 2 starting point
data = pd.read_csv(TRAIN_CSV)
print(data)

# --- Phase 2, Cell 2 ---
# Check the schema — column names, data types, and non-null counts
print(data.info())

# --- Phase 2, Cell 3 ---
# Look for any missing values across every column
print(data.isnull().sum())

# --- Phase 2, Cell 4 ---
# Descriptive statistics: mean, std, min, max, quartiles for all numeric columns
print(data.describe())

# --- Phase 2, Cell 5 ---
# Count how many customers fall under each gender category
print(data['Gender'].value_counts())

# --- Phase 2, Cell 6 ---
# Break down customers by subscription tier (Basic / Standard / Premium)
print(data['Subscription Type'].value_counts())

# --- Phase 2, Cell 7 ---
# Break down customers by contract length (Monthly / Quarterly / Annual)
print(data['Contract Length'].value_counts())

# --- Phase 2, Cell 8 ---
# Drop the CustomerID column — it is just an identifier, not a predictive feature
data = data.drop('CustomerID', axis=1)
print('CustomerID column dropped.')

# --- Phase 2, Cell 9 ---
# Double-check for any remaining null entries after the drop
print(data.isnull())

# ── Phase 2 Visualization — 6 charts from the Phase 2 report ──
# All charts are saved as PNG files to backend/eda_plots/ rather than shown interactively.

# Reload with CustomerID still present so charts are computed on the clean full frame
df_viz = pd.read_csv(TRAIN_CSV)

# Figure 1 — Bar chart: churned vs retained count split by subscription type
fig, ax = plt.subplots(figsize=(8, 5))
for churn_val, label, color in [(0, 'Retained', '#2E7D32'), (1, 'Churned', '#C62828')]:
    sub_counts = df_viz[df_viz['Churn'] == churn_val]['Subscription Type'].value_counts()
    ax.bar(
        [f"{t}\n({label})" for t in sub_counts.index],
        sub_counts.values, color=color, alpha=0.85, label=label
    )
ax.set_title('Figure 1: Churn Count by Subscription Type')
ax.set_ylabel('Number of Customers')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "fig1_churn_by_subscription.png"), dpi=120)
plt.close()
print("[Phase 2] Saved fig1_churn_by_subscription.png")

# Figure 2 — Histogram: age distribution across all customers
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df_viz['Age'], bins=20, color='#1F4E79', alpha=0.8, edgecolor='white')
ax.set_title('Figure 2: Age Distribution of Customers')
ax.set_xlabel('Age')
ax.set_ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "fig2_age_distribution.png"), dpi=120)
plt.close()
print("[Phase 2] Saved fig2_age_distribution.png")

# Figure 3 — Box plot: total spend comparison between churned and retained customers
churned     = df_viz[df_viz['Churn'] == 1]['Total Spend']
not_churned = df_viz[df_viz['Churn'] == 0]['Total Spend']
fig, ax = plt.subplots(figsize=(7, 5))
ax.boxplot(
    [not_churned, churned],
    patch_artist=True,
    boxprops=dict(facecolor='#CFE0F0', color='#1F4E79')
)
ax.set_xticks([1, 2])
ax.set_xticklabels(['Retained', 'Churned'])
ax.set_title('Figure 3: Total Spend by Churn Status')
ax.set_ylabel('Total Spend ($)')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "fig3_spend_by_churn.png"), dpi=120)
plt.close()
print("[Phase 2] Saved fig3_spend_by_churn.png")

# Figure 4 — Scatter plot: tenure vs total spend, points coloured by churn status
sample = df_viz.sample(min(3000, len(df_viz)), random_state=42)
fig, ax = plt.subplots(figsize=(8, 5))
for churn_val, label, color in [(0, 'Retained', '#2E7D32'), (1, 'Churned', '#C62828')]:
    sub = sample[sample['Churn'] == churn_val]
    ax.scatter(sub['Tenure'], sub['Total Spend'], alpha=0.3, label=label, color=color, s=12)
ax.set_title('Figure 4: Tenure vs Total Spend (coloured by Churn)')
ax.set_xlabel('Tenure (months)')
ax.set_ylabel('Total Spend ($)')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "fig4_tenure_vs_spend.png"), dpi=120)
plt.close()
print("[Phase 2] Saved fig4_tenure_vs_spend.png")

# Figure 5 — Heatmap: pairwise Pearson correlation between all numerical features
numeric_cols = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls',
                'Payment Delay', 'Total Spend', 'Last Interaction', 'Churn']
corr = df_viz[numeric_cols].corr()
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(range(len(numeric_cols)))
ax.set_yticks(range(len(numeric_cols)))
ax.set_xticklabels(numeric_cols, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(numeric_cols, fontsize=8)
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center', fontsize=7)
plt.colorbar(im, ax=ax)
ax.set_title('Figure 5: Pearson Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "fig5_correlation_heatmap.png"), dpi=120)
plt.close()
print("[Phase 2] Saved fig5_correlation_heatmap.png")

# Figure 6 — Pie chart: overall proportion of churned vs retained customers
fig, ax = plt.subplots(figsize=(6, 6))
counts = df_viz['Churn'].value_counts()
ax.pie(counts.values, labels=['Retained', 'Churned'], autopct='%1.1f%%',
       colors=['#2E7D32', '#C62828'], startangle=140)
ax.set_title('Figure 6: Overall Churn Proportion')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "fig6_churn_pie.png"), dpi=120)
plt.close()
print("[Phase 2] Saved fig6_churn_pie.png")


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 3 — Machine Learning Model Development & Evaluation
#  (Classification: Logistic Regression, Decision Tree, Random Forest
#   Regression: Linear Regression, Random Forest Regressor for Total Spend
#   — plus the supplementary Support Calls → Payment Delay regression snippet)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("PHASE 3 — Model Development & Evaluation")
print("=" * 60)

# ── Phase 3, Step 1 — Load and consolidate both CSV files into one dataset ──
# The Phase 3 report merged training + testing into 442,211 unique records
# for more robust model training
train_raw = pd.read_csv(TRAIN_CSV).dropna()
test_raw  = pd.read_csv(TEST_CSV).dropna()

# Concatenate, remove fully-duplicate rows, then deduplicate by CustomerID
combined = pd.concat([train_raw, test_raw], ignore_index=True)
combined = combined.drop_duplicates()
combined = combined.drop_duplicates(subset=['CustomerID'], keep='first')

print("Combined dataset shape:", combined.shape)
print("Overall churn rate:", combined['Churn'].mean())

# ── Phase 3, Step 2 — Encode the three categorical columns ──
# LabelEncoder converts Gender, Subscription Type, and Contract Length
# from text strings into integer codes that scikit-learn can process
df = combined.copy()

le_gender   = LabelEncoder()
le_sub      = LabelEncoder()
le_contract = LabelEncoder()

df['Gender']            = le_gender.fit_transform(df['Gender'])
df['Subscription Type'] = le_sub.fit_transform(df['Subscription Type'])
df['Contract Length']   = le_contract.fit_transform(df['Contract Length'])

# ── Phase 3, Step 3 — Define the 10 predictive features and target variable ──
feat_cols_clf = [
    'Age', 'Gender', 'Tenure', 'Usage Frequency', 'Support Calls',
    'Payment Delay', 'Subscription Type', 'Contract Length',
    'Total Spend', 'Last Interaction'
]

X = df[feat_cols_clf]
y = df['Churn'].astype(int)

# Stratified 80/20 split — preserves the churn ratio in both training and test partitions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Phase 3, Scaling — Logistic Regression needs features on a common scale ──
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── Phase 3, Step 4 — Train all three classification models ──

# Logistic Regression is a linear baseline — quick to train, interpretable
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)

# Decision Tree can capture non-linear patterns without needing scaled features
dt = DecisionTreeClassifier(max_depth=8, random_state=42)
dt.fit(X_train, y_train)

# Random Forest — an ensemble of 150 decision trees that outperformed all baselines
rf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# ── Phase 3, Step 5 — Get predictions and compare all three classifiers ──
pred_lr  = lr.predict(X_test_scaled)
pred_dt  = dt.predict(X_test)
pred_rf  = rf.predict(X_test)

classifiers = {
    'Logistic Regression': pred_lr,
    'Decision Tree':       pred_dt,
    'Random Forest':       pred_rf,
}

print("\n=== CLASSIFICATION RESULTS ===")
for name, pred in classifiers.items():
    print(f"\n--- {name} ---")
    print("Accuracy: ",  round(accuracy_score(y_test, pred),  4))
    print("Precision:", round(precision_score(y_test, pred),  4))
    print("Recall:   ",  round(recall_score(y_test, pred),    4))
    print("F1-Score: ",  round(f1_score(y_test, pred),        4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, target_names=['No Churn', 'Churn']))

# Print which features the Random Forest found most important for predicting churn
importances_clf = pd.Series(rf.feature_importances_, index=feat_cols_clf)
print("\nFeature Importance — Churn Classification:")
print(importances_clf.sort_values(ascending=False))

# ── Phase 3 — Regression task: predict Total Spend from behavioral features ──
# This was run as a supporting regression analysis alongside the main classification task

feat_cols_reg = [
    'Age', 'Gender', 'Tenure', 'Usage Frequency', 'Support Calls',
    'Payment Delay', 'Subscription Type', 'Contract Length',
    'Last Interaction', 'Churn'
]

Xr = df[feat_cols_reg]
yr = df['Total Spend']

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    Xr, yr, test_size=0.2, random_state=42
)

# Scale the regression features for the Linear Regression model
scaler_r = StandardScaler()
scaler_r.fit(Xr_train)
Xr_train_scaled = scaler_r.transform(Xr_train)
Xr_test_scaled  = scaler_r.transform(Xr_test)

# Linear Regression — simple, interpretable spend prediction
lin_reg = LinearRegression()
lin_reg.fit(Xr_train_scaled, yr_train)
pred_lin = lin_reg.predict(Xr_test_scaled)

# Random Forest Regressor — non-linear spend prediction
rfreg = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
rfreg.fit(Xr_train, yr_train)
pred_rfreg = rfreg.predict(Xr_test)

regressors = {
    'Linear Regression':       pred_lin,
    'Random Forest Regressor': pred_rfreg,
}

print("\n\n=== REGRESSION RESULTS (Target: Total Spend) ===")
for name, pred in regressors.items():
    mae  = mean_absolute_error(yr_test, pred)
    mse  = mean_squared_error(yr_test, pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(yr_test, pred)
    print(f"\n--- {name} ---")
    print(f"MAE:  ${mae:.2f}")
    print(f"MSE:  {mse:.2f}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"R2:   {r2:.4f}")

importances_reg = pd.Series(rfreg.feature_importances_, index=feat_cols_reg)
print("\nFeature Importance — Total Spend Regression:")
print(importances_reg.sort_values(ascending=False))

# ── Phase 3 — Supplementary snippet: Support Calls → Payment Delay regression ──
# This exact snippet was included in the project notebooks to explore whether
# the number of support calls could predict how many days a payment was delayed

print("\n" + "─" * 55)
print("Phase 3 — Supplementary: Support Calls → Payment Delay")
print("─" * 55)

# The supplementary snippet reads from the testing CSV as originally written
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

data = pd.read_csv(TEST_CSV)
data
x = data["Support Calls"].values.reshape(-1, 1)
y = data["Payment Delay"].values.reshape(-1, 1)
print(x)
print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
model = LinearRegression()
model.fit(x_train, y_train)
new = model.predict(y_test)
print(new)

mae  = mean_absolute_error(y_test, new)
print(mae)
mse  = mean_squared_error(y_test, new)
print(mse)
rmse = np.sqrt(mse)
print(rmse)

# Save the regression scatter plot instead of calling plt.show() (no display available)
plt.figure(figsize=(12, 6))
plt.plot(x, y, 'ro', markersize=2, alpha=0.3)
plt.plot(y_test, new)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Support Calls vs Payment Delay — Linear Regression')
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "phase3_supp_regression.png"), dpi=120)
plt.close()
print("[Phase 3] Saved phase3_supp_regression.png")

# --- Phase 2 EDA cells (original Jupyter notebook form, re-included here) ---
# These 9 cells mirror the exact sequence from the Phase 2 report

# --- Cell 1 ---
import pandas as pd
data = pd.read_csv(TRAIN_CSV)
print(data)

# --- Cell 2 ---
print(data.info())

# --- Cell 3 ---
print(data.isnull().sum())

# --- Cell 4 ---
print(data.describe())

# --- Cell 5 ---
print(data['Gender'].value_counts())

# --- Cell 6 ---
print(data['Subscription Type'].value_counts())

# --- Cell 7 ---
print(data['Contract Length'].value_counts())

# --- Cell 8 ---
data = data.drop('CustomerID', axis=1)
print('CustomerID column dropped.')

# --- Cell 9 ---
data.isnull()


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 4 — Final Model Selection, Serialization & Deployment Prep
#  (The Random Forest is retrained on the full consolidated dataset,
#   then saved to disk so server.py can load it and serve predictions
#   through the web interface without retraining on every request.)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("PHASE 4 — Final Model Serialization & Deployment")
print("=" * 60)

# Re-encode from the combined dataset using separate encoder objects
# (these will be serialized and loaded by server.py at runtime)
df_final = combined.copy()

le_gender_final   = LabelEncoder()
le_sub_final      = LabelEncoder()
le_contract_final = LabelEncoder()

df_final['Gender_enc']            = le_gender_final.fit_transform(df_final['Gender'])
df_final['Subscription Type_enc'] = le_sub_final.fit_transform(df_final['Subscription Type'])
df_final['Contract Length_enc']   = le_contract_final.fit_transform(df_final['Contract Length'])

# These are the exact column names the prediction API expects at inference time
feat_cols_final = [
    'Age', 'Gender_enc', 'Tenure', 'Usage Frequency', 'Support Calls',
    'Payment Delay', 'Subscription Type_enc', 'Contract Length_enc',
    'Total Spend', 'Last Interaction'
]

X_final = df_final[feat_cols_final]
y_final = df_final['Churn'].astype(int)

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_final, y_final, test_size=0.2, random_state=42, stratify=y_final
)

# Train the production Random Forest — same hyperparameters as Phase 3 winner
final_model = RandomForestClassifier(
    n_estimators=150, max_depth=10, random_state=42, n_jobs=-1
)
final_model.fit(X_train_f, y_train_f)

# Evaluate the final model one last time on the hold-out partition
pred_final = final_model.predict(X_test_f)
print("Final Model — Random Forest Classifier")
print("Accuracy: ",  round(accuracy_score(y_test_f,  pred_final), 4))
print("Precision:", round(precision_score(y_test_f, pred_final), 4))
print("Recall:   ",  round(recall_score(y_test_f,    pred_final), 4))
print("F1-Score: ",  round(f1_score(y_test_f,        pred_final), 4))

# Serialize the trained model and all encoder objects to the models/ folder
joblib.dump(final_model,       os.path.join(MODELS_DIR, "churn_model.pkl"))
joblib.dump(le_gender_final,   os.path.join(MODELS_DIR, "le_gender.pkl"))
joblib.dump(le_sub_final,      os.path.join(MODELS_DIR, "le_sub.pkl"))
joblib.dump(le_contract_final, os.path.join(MODELS_DIR, "le_contract.pkl"))
joblib.dump(feat_cols_final,   os.path.join(MODELS_DIR, "feat_cols.pkl"))

# Save a metadata JSON file that server.py reads for the /api/model-info endpoint
importances_final = pd.Series(
    final_model.feature_importances_, index=feat_cols_final
)
metadata = {
    "algorithm":    "Random Forest Classifier",
    "n_estimators": 150,
    "max_depth":    10,
    "random_state": 42,
    "accuracy":     round(accuracy_score(y_test_f,  pred_final), 4),
    "precision":    round(precision_score(y_test_f, pred_final), 4),
    "recall":       round(recall_score(y_test_f,    pred_final), 4),
    "f1":           round(f1_score(y_test_f,        pred_final), 4),
    "records_trained": int(len(df_final)),
    "feature_importances": {
        col: round(float(imp), 4)
        for col, imp in importances_final.sort_values(ascending=False).items()
    }
}
with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("\nModel and encoders saved to:", os.path.abspath(MODELS_DIR))
print("Gender classes:       ", le_gender_final.classes_)
print("Subscription classes: ", le_sub_final.classes_)
print("Contract classes:     ", le_contract_final.classes_)
print("Metadata written to metadata.json")

# ── Phase 4 — Sanity check with the two documented demo profiles ──
# These are the exact two customers shown in the Phase 4 screenshots

# High-risk profile: 58yo Male, Basic Monthly, 9 months tenure, 9 calls, 27-day delay
high_risk_row = pd.DataFrame([[
    58,
    int(le_gender_final.transform(["Male"])[0]),
    9, 6, 9, 27,
    int(le_sub_final.transform(["Basic"])[0]),
    int(le_contract_final.transform(["Monthly"])[0]),
    210, 28
]], columns=feat_cols_final)
hr_prob = final_model.predict_proba(high_risk_row)[0][1]
print(f"\n[Sanity check] High-Risk demo → Churn probability: {hr_prob * 100:.1f}%")

# Low-risk profile: 34yo Female, Premium Annual, 48 months tenure, 1 call, 2-day delay
low_risk_row = pd.DataFrame([[
    34,
    int(le_gender_final.transform(["Female"])[0]),
    48, 26, 1, 2,
    int(le_sub_final.transform(["Premium"])[0]),
    int(le_contract_final.transform(["Annual"])[0]),
    890, 3
]], columns=feat_cols_final)
lr_prob = final_model.predict_proba(low_risk_row)[0][1]
print(f"[Sanity check] Low-Risk  demo → Churn probability: {lr_prob * 100:.1f}%")

print("\n✓ All phases complete. Run server.py to start the web API.")


if __name__ == "__main__":
    # Running this script directly will execute the full pipeline:
    # Phase 1 → data load, Phase 2 → EDA + charts, Phase 3 → model evaluation,
    # Phase 4 → serialize final model and encoders ready for server.py to load
    pass
