# 🚀 E-commerce Customer Churn Prediction

**Production-Grade MLOps Implementation** | XGBoost · CatBoost · LightGBM · MLflow · DVC · FastAPI

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.9%2B-orange)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-3.30%2B-purple)](https://dvc.org/)

---

## 🎯 Project Overview

A **brutally honest, industry-standard** machine learning system for predicting e-commerce customer churn. This project uses **Gradient Boosted Trees** (XGBoost, CatBoost, LightGBM) instead of unnecessary deep learning because churn prediction is a **tabular data problem**.

### Why This Tech Stack?

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Models** | XGBoost/CatBoost/LightGBM | Faster & more accurate than deep learning for tabular data |
| **Experiment Tracking** | MLflow | Track every experiment, compare models, manage model registry |
| **Data Versioning** | DVC | Version control for datasets (Git for data) |
| **API** | FastAPI | Modern, fast, automatic Swagger docs (Flask is outdated) |
| **Monitoring** | Evidently AI | Detect data drift and model degradation |
| **Deployment** | Docker | "Works on my machine" is not an excuse |

---

## 📊 Dataset

**Source:** [Kaggle - E-commerce Customer Churn](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction) by Ankit Verma

**Features:**
- **Customer Demographics:** Gender, Marital Status, City Tier
- **Behavioral Metrics:** Tenure, Order Count, Days Since Last Order
- **Engagement:** Hours on App, Satisfaction Score, Complaints
- **Financial:** Cashback Amount, Coupon Usage, Order Amount Hike

---

## 🏗️ Project Structure

```
ecommerce_customer_churn/
├── app/
│   ├── fastapi_app.py          # FastAPI application
│   └── streamlit_app.py         # Streamlit dashboard
├── src/ecommerce_customer_churn/
│   ├── components/
│   │   ├── data_ingestion.py       # Kaggle API integration
│   │   ├── data_transformation.py  # Feature engineering + SMOTE
│   │   ├── model_trainer.py        # XGBoost/CatBoost/LightGBM training
│   │   ├── model_evalution.py      # Comprehensive evaluation
│   │   └── model_monitoring.py     # Evidently AI drift detection
│   ├── pipelines/
│   │   ├── training_pipeline.py    # End-to-end training
│   │   └── prediction_pipeline.py  # Inference pipeline
│   ├── exception.py                # Custom exception handling
│   ├── logger.py                   # Professional logging (loguru)
│   └── utils.py                    # Helper functions
├── data/
│   ├── raw/                    # Raw data (tracked by DVC)
│   └── processed/              # Processed data
├── models/                     # Trained models
├── artifacts/                  # Plots, reports, preprocessors
├── logs/                       # Application logs
├── params.yaml                 # Centralized configuration
├── dvc.yaml                    # DVC pipeline definition
├── requirements.txt            # Production dependencies
├── setup.py                    # Package setup
├── Dockerfile                  # Docker configuration
└── run_pipeline.py             # Training entry point
```

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/ecommerce_customer_churn.git
cd ecommerce_customer_churn
```

### 2. Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Kaggle API

```bash
# Create .kaggle directory
mkdir ~/.kaggle

# Copy your kaggle.json
cp kaggle.json ~/.kaggle/

# Set permissions (Linux/Mac)
chmod 600 ~/.kaggle/kaggle.json
```

### 4. Train Model

```bash
python run_pipeline.py
```

This will:
- ✅ Download data from Kaggle
- ✅ Perform feature engineering (RFM features, engagement scores)
- ✅ Handle class imbalance with SMOTE
- ✅ Train XGBoost, CatBoost, and LightGBM
- ✅ Log experiments to MLflow
- ✅ Save best model based on ROC-AUC
- ✅ Generate evaluation plots (confusion matrix, ROC curve, SHAP values)

### 5. View MLflow Experiments

```bash
mlflow ui
```

Open http://localhost:5000 to compare model runs.

### 6. Run FastAPI Server

```bash
uvicorn app.fastapi_app:app --reload
```

Open http://localhost:8000/docs for Swagger UI.

### 7. Run Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## 🔧 MLOps Features

### Experiment Tracking (MLflow)

```python
# Automatically logs:
- Model parameters
- Evaluation metrics (ROC-AUC, F1, Precision, Recall)
- Feature importance
- Model artifacts
- SHAP values
```

### Data Versioning (DVC)

```bash
# Initialize DVC
dvc init

# Add remote storage (DagsHub)
dvc remote add -d origin https://dagshub.com/username/ecommerce_customer_churn.dvc

# Track data
dvc add data/raw/ecommerce_churn.csv
git add data/raw/ecommerce_churn.csv.dvc
git commit -m "Track raw data"

# Push data to remote
dvc push
```

### Model Monitoring (Evidently AI)

```python
# Detect data drift
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=train_df, current_data=new_data)
report.save_html("reports/drift_report.html")
```

---

## 📡 API Usage

### Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Tenure": 10,
    "CityTier": 1,
    "WarehouseToHome": 15.0,
    "HourSpendOnApp": 3.0,
    "NumberOfDeviceRegistered": 3,
    "SatisfactionScore": 3,
    "NumberOfAddress": 2,
    "Complain": 0,
    "OrderAmountHikeFromlastYear": 15.0,
    "CouponUsed": 5,
    "OrderCount": 10,
    "DaySinceLastOrder": 5,
    "CashbackAmount": 150.0,
    "PreferredLoginDevice": "Mobile Phone",
    "PreferredPaymentMode": "Debit Card",
    "Gender": "Male",
    "PreferedOrderCat": "Laptop & Accessory",
    "MaritalStatus": "Single"
  }'
```

### Batch Prediction

```bash
curl -X POST "http://localhost:8000/predict_csv" \
  -F "file=@customers.csv"
```

---

## 🐳 Docker Deployment

```bash
# Build image
docker build -t ecommerce-churn-api .

# Run container
docker run -p 8000:8000 ecommerce-churn-api
```

---

## 📈 Model Performance

| Model | ROC-AUC | F1-Score | Precision | Recall |
|-------|---------|----------|-----------|--------|
| XGBoost | 0.XX | 0.XX | 0.XX | 0.XX |
| CatBoost | 0.XX | 0.XX | 0.XX | 0.XX |
| LightGBM | 0.XX | 0.XX | 0.XX | 0.XX |

*(Run training to populate metrics)*

---

## 🛠️ Tech Stack Summary

```
┌─────────────────────────────────────────────┐
│  PRODUCTION-GRADE ML STACK                  │
├─────────────────────────────────────────────┤
│  Data Ingestion    → Kaggle API             │
│  Feature Eng       → RFM, Engagement Score  │
│  Class Imbalance   → SMOTE                  │
│  Models            → XGBoost/CatBoost/LGBM  │
│  Tracking          → MLflow                 │
│  Versioning        → DVC + Git              │
│  API               → FastAPI                │
│  Frontend          → Streamlit              │
│  Monitoring        → Evidently AI           │
│  Deployment        → Docker                 │
└─────────────────────────────────────────────┘
```

---

## 🤝 Contributing

This is a portfolio project, but suggestions are welcome!

---

## 📝 License

MIT License

---

## 👨‍💻 Author

**rkpcode**

---

## 🎓 Learning Resources

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [DVC Get Started](https://dvc.org/doc/start)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)

---

**⚠️ Remember:** Churn prediction is a **tabular data problem**. Don't use deep learning just because it sounds cool. Gradient Boosting is faster, more accurate, and requires less data.
#   c u s t o m e r _ c h u r n _ p r e d i c t i o n  
 