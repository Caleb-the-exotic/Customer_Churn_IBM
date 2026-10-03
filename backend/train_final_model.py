"""
Final model training script for Phase 4 deployment.
Trains the best-performing model (Random Forest Classifier) on the
consolidated dataset and saves it along with encoders/scaler for
use in the desktop application.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---- Load & consolidate ----
train_raw = pd.read_csv('/mnt/user-data/uploads/customer_churn_dataset-training-master.csv').dropna()
test_raw = pd.read_csv('/mnt/user-data/uploads/customer_churn_dataset-testing-master.csv').dropna()

combined = pd.concat([train_raw, test_raw], ignore_index=True)
combined = combined.drop_duplicates()
combined = combined.drop_duplicates(subset=['CustomerID'], keep='first')

df = combined.copy()

# ---- Encode categoricals ----
le_gender = LabelEncoder()
le_sub = LabelEncoder()
le_contract = LabelEncoder()

df['Gender_enc'] = le_gender.fit_transform(df['Gender'])
df['Subscription Type_enc'] = le_sub.fit_transform(df['Subscription Type'])
df['Contract Length_enc'] = le_contract.fit_transform(df['Contract Length'])

feat_cols = ['Age', 'Gender_enc', 'Tenure', 'Usage Frequency', 'Support Calls',
             'Payment Delay', 'Subscription Type_enc', 'Contract Length_enc',
             'Total Spend', 'Last Interaction']

X = df[feat_cols]
y = df['Churn'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- Train final model ----
final_model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
final_model.fit(X_train, y_train)

pred = final_model.predict(X_test)
print("Final Model — Random Forest Classifier")
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("Precision:", round(precision_score(y_test, pred), 4))
print("Recall:", round(recall_score(y_test, pred), 4))
print("F1-Score:", round(f1_score(y_test, pred), 4))

# ---- Save model + encoders for deployment ----
joblib.dump(final_model, '/home/claude/app/churn_model.pkl')
joblib.dump(le_gender, '/home/claude/app/le_gender.pkl')
joblib.dump(le_sub, '/home/claude/app/le_sub.pkl')
joblib.dump(le_contract, '/home/claude/app/le_contract.pkl')
joblib.dump(feat_cols, '/home/claude/app/feat_cols.pkl')

print("\nModel and encoders saved to /home/claude/app/")
print("Gender classes:", le_gender.classes_)
print("Subscription classes:", le_sub.classes_)
print("Contract classes:", le_contract.classes_)
