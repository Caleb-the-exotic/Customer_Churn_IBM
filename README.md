# Customer Churn Prediction System — TeleSaaS Analytics

A modern, production-grade Customer Churn Prediction web application for subscription and telecom businesses. The frontend is built with pure HTML5, Tailwind CSS, Vanilla JavaScript, Chart.js, Papa Parse, and Lucide icons, connecting directly to a trained Scikit-Learn **Random Forest Classifier** (`churn_model.pkl`, **99.35% test accuracy**) served via a Flask REST API.

---

## 1. Quick Start / How to Launch Locally

### Prerequisites
- Python 3.10+ (with `pandas`, `scikit-learn`, `joblib`, `flask`, `flask-cors`)

### Step 1: Start the Python Backend Server
From the root workspace directory (`Churn_app/`):
```bash
python server.py
```
The server will start at `http://127.0.0.1:5000` and load the trained model artifacts and encoders from `models/`.

### Step 2: Open the Web Application
Open your browser and navigate to:
```
http://127.0.0.1:5000
```
*(The Flask server directly hosts `index.html`, `style.css`, and `script.js`).*

---

## 2. Why Do Many Predictions Return > 99% Churn Probability?

The application is connected directly to the real trained **Random Forest Classifier (150 estimators, max depth 10)** trained on the consolidated **442,211 customer records** from `backend/customer_churn_dataset-training-master.csv` and `backend/customer_churn_dataset-testing-master.csv`.

In this dataset, the underlying business rules and data generation patterns establish:
1. **Contract Length = Monthly**: In the training dataset of 442,211 customers, **100% of Monthly contract customers churned (87,104 out of 87,104, exactly 0 retained!)**. When Contract Length is "Monthly", almost all 150 decision trees vote for Churn = 1 (giving > 99% churn probability).
2. **Support Calls >= 5**: In the training dataset, customers with 5 or more support calls churned at near 100%. Support Calls holds the highest feature importance (**32.7%**).
3. **Payment Delay >= 20 days**: Positive association with churn; extreme delays indicate pending cancellation.
4. **Total Spend < $500**: Retained customers in the training data had Total Spend >= $500.
5. **Age > 50**: Older customers in the dataset exhibited near-complete churn.

### How to See Low-Risk Predictions (< 5%)
- Click the **"Load Low-Risk Demo"** button on the **Predict Churn** page:
  - Age: `34`
  - Gender: `Female`
  - Tenure: `48 months`
  - Usage Frequency: `26/month`
  - Support Calls: `1`
  - Payment Delay: `2 days`
  - Subscription Type: `Premium`
  - Contract Length: `Annual`
  - Total Spend: `$890`
  - Last Interaction: `3 days`
- Click **"Predict Churn Risk"**.
- Result: **3.58% Churn Probability (Low Risk / Likely to Stay)**.

---

## 3. Application Structure

```
Churn_app/
├── backend/                                   # Legacy backend files (Unmodified)
│   ├── churn_app.py                           # Legacy desktop Tkinter app (superseded by web UI)
│   ├── train_final_model.py                   # Phase 4 training script
│   ├── customer_churn_dataset-training-master.csv # 440,833 records
│   └── customer_churn_dataset-testing-master.csv  # 64,374 records
├── models/                                    # Serialized Model Artifacts
│   ├── churn_model.pkl                        # Random Forest (150 estimators, max_depth 10)
│   ├── le_gender.pkl                          # LabelEncoder for Gender
│   ├── le_sub.pkl                             # LabelEncoder for Subscription Type
│   ├── le_contract.pkl                        # LabelEncoder for Contract Length
│   ├── feat_cols.pkl                          # 10 feature names in exact column order
│   └── metadata.json                          # Hyperparameters, evaluation metrics, feature importances
├── index.html                                 # Single-page application structure (6 active views)
├── style.css                                  # Custom CSS tokens, badges, and animation styles
├── script.js                                  # Modular state management, Chart.js, Papa Parse, REST API client
├── server.py                                  # Flask REST API server & static asset host
└── README.md                                  # Documentation and run instructions
```

---

## 4. Expected Python Backend Endpoints

### 1. `GET /api/health`
Checks server status, model readiness, and dataset benchmark size.
- **Response**:
```json
{
  "status": "online",
  "model_loaded": true,
  "model_name": "Random Forest Classifier",
  "dataset_records_trained": 442211,
  "reported_accuracy": 0.9935
}
```

### 2. `GET /api/model-info`
Returns algorithm specifications, comparison table benchmarks, and regression analysis data.

### 3. `POST /api/predict`
Calculates real-time churn risk for an individual customer.
- **Request**:
```json
{
  "Age": 58,
  "Gender": "Male",
  "Tenure": 9,
  "Usage Frequency": 6,
  "Support Calls": 9,
  "Payment Delay": 27,
  "Subscription Type": "Basic",
  "Contract Length": "Monthly",
  "Total Spend": 210,
  "Last Interaction": 28
}
```
- **Response**:
```json
{
  "prediction": 1,
  "churn_probability": 0.9998,
  "risk_category": "High Risk",
  "model_name": "Random Forest Classifier",
  "reported_accuracy": 0.9935,
  "is_real_backend": true,
  "risk_factors": ["Elevated support call volume (9 calls)", "..."],
  "recommendations": ["Conduct an urgent customer success follow-up call...", "..."]
}
```

### 4. `POST /api/batch-predict`
Runs inference on an array of customer records.
- **Request**: `{ "records": [ { "CustomerID": "CUST-001", "Age": ..., ... } ] }`
- **Response**: Batch summary statistics and an array of individual prediction outputs.

### 5. `GET /api/sample-dataset?limit=300`
Streams real customer records from `backend/customer_churn_dataset-testing-master.csv` into the Customer Explorer table.

---

## 5. Feature Matrix

| Feature | Works Offline (Standalone Frontend) | Requires Backend Integration (`server.py`) |
| :--- | :---: | :---: |
| **Overview Dashboard & KPIs** | Yes (Phase 2 Benchmark dataset & calculations) | Yes (when loading live server benchmark) |
| **Interactive Chart.js Visualizations** | Yes (Responsive client-side rendering) | — |
| **Single Customer Prediction Form** | Yes (Fallback EDA heuristic with simulation warning) | **Yes (Real Random Forest ML Model Inference)** |
| **Customer Explorer Table** | Yes (Filtering, sorting, search, CSV export) | Yes (to stream real testing sample rows) |
| **Churn Analytics** | Yes (Multi-cohort rate analysis) | — |
| **Model Performance Benchmarks** | Yes (Reported Phase 3/4 evaluation metrics) | Yes (live hyperparameter audit) |
| **CSV Drag-and-Drop & Quality Audit** | Yes (Papa Parse browser-side schema validator) | — |
| **Batch Prediction Engine** | Yes (Heuristic mode) | **Yes (Fast vectorized ML inference)** |
| **CSV Export of Results** | Yes (Client-side Blob download) | — |
