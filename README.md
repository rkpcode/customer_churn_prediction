# 🎯 E-commerce Customer Churn Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-LightGBM-green)](https://lightgbm.readthedocs.io/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-red)](https://streamlit.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-orange)](https://shap.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> An end-to-end machine learning solution for predicting customer churn in e-commerce platforms with explainable AI capabilities.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Architecture](#project-architecture)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a production-ready machine learning pipeline to predict customer churn in e-commerce businesses. By identifying customers at risk of churning, businesses can take proactive retention measures and reduce customer attrition.

### Business Impact

- **99.91% ROC-AUC**: Exceptional model discrimination capability
- **98.42% Recall**: Catches almost all potential churners
- **92.57% Precision**: Efficient targeting with minimal false positives
- **Explainable AI**: SHAP integration for transparent decision-making

## ✨ Key Features

### 🔬 Machine Learning Pipeline

- **Data Ingestion**: Automated data download from Kaggle API
- **Data Transformation**: 
  - Stratified train-test split (80-20)
  - Missing value imputation with signal preservation
  - Label encoding for categorical features
  - Phase-based feature engineering
- **Model Training**: 
  - Baseline-first approach (Dumb → Logistic Regression → Tree Ensembles)
  - Multiple model comparison (XGBoost, LightGBM, CatBoost, Random Forest, Gradient Boosting)
  - Business-aligned threshold tuning
  - Comprehensive evaluation metrics

### 🎨 Interactive Web Application

- **Single Prediction**: Real-time churn risk assessment for individual customers
- **Batch Prediction**: Bulk processing via CSV upload
- **Model Insights**: Performance metrics and model comparison
- **Explainable AI**: SHAP waterfall plots and global feature importance

### 🔍 Explainability

- **SHAP Integration**: Understand which features drive predictions
- **Waterfall Plots**: Visualize feature contributions for individual predictions
- **Global Feature Importance**: Identify key churn indicators across all customers

## 🏗️ Project Architecture

```
┌─────────────────┐
│  Data Ingestion │
│   (Kaggle API)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Transform  │
│  - Train/Test   │
│  - Feature Eng  │
│  - Encoding     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Training  │
│  - 7 Models     │
│  - Threshold    │
│  - Evaluation   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Streamlit App  │
│  + SHAP Explai  │
└─────────────────┘
```

## 📊 Model Performance

### Model Comparison

| Model | ROC-AUC | Recall | Precision | F1-Score |
|-------|---------|--------|-----------|----------|
| **LightGBM** ⭐ | **0.9991** | **98.42%** | **92.57%** | **0.9541** |
| XGBoost | 0.9989 | 100.00% | 93.14% | 0.9645 |
| CatBoost | 0.9955 | 98.95% | 84.30% | 0.9104 |
| Random Forest | 0.9976 | 87.89% | 95.98% | 0.9176 |
| Gradient Boosting | 0.9431 | 64.74% | 80.39% | 0.7172 |
| Logistic Regression | 0.8782 | 83.68% | 44.29% | 0.5792 |
| Dumb Baseline | 0.5000 | 0.00% | 0.00% | 0.0000 |

### Threshold Tuning

- **Strategy**: Target top 20% highest-risk customers
- **Optimized Threshold**: 0.247 (vs default 0.5)
- **Result**: 99.47% recall with 83.63% precision

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Kaggle API credentials (for data download)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/rkpcode/customer_churn_prediction.git
cd customer_churn_prediction
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure Kaggle API**
```bash
# Place your kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\Users\<username>\.kaggle\ (Windows)
```

## 💻 Usage

### Training the Pipeline

Run the complete ML pipeline:

```bash
python run_pipeline.py
```

Or run individual components:

```bash
# Data Ingestion
python -m src.ecommerce_customer_churn.components.data_ingestion

# Data Transformation
python -m src.ecommerce_customer_churn.components.data_transformation

# Model Training
python -m src.ecommerce_customer_churn.components.model_trainer
```

### Running the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

The app will be available at `http://localhost:8501`

### Making Predictions

**Single Prediction:**
1. Open the Streamlit app
2. Navigate to "Single Prediction" tab
3. Fill in customer details
4. Click "Predict Churn Risk with Explanation"
5. View prediction with SHAP explanation

**Batch Prediction:**
1. Navigate to "Batch Prediction" tab
2. Download sample CSV template
3. Upload your customer data CSV
4. Click "Run Batch Prediction"
5. Download results with churn probabilities

## 📁 Project Structure

```
ecommerce_customer_churn/
│
├── app/
│   ├── streamlit_app.py          # Streamlit web application
│   ├── fastapi_app.py             # FastAPI REST API
│   └── flask_app.py               # Flask web application
│
├── src/ecommerce_customer_churn/
│   ├── components/
│   │   ├── data_ingestion.py     # Data download and validation
│   │   ├── data_transformation.py # Feature engineering
│   │   ├── model_trainer.py      # Model training and evaluation
│   │   ├── model_evaluation.py   # Model evaluation utilities
│   │   └── model_monitoring.py   # Model monitoring
│   │
│   ├── pipelines/
│   │   ├── training_pipeline.py  # End-to-end training pipeline
│   │   └── prediction_pipeline.py # Prediction pipeline
│   │
│   ├── utils.py                   # Utility functions
│   ├── logger.py                  # Logging configuration
│   └── exception.py               # Custom exceptions
│
├── notebooks/
│   ├── 01_eda_industry_grade.ipynb      # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb     # Feature Engineering
│   └── 03_model_training.ipynb          # Model Training & Evaluation
│
├── data/
│   ├── raw/                       # Raw data from Kaggle
│   └── processed/                 # Processed feature sets
│
├── models/
│   ├── best_model.pkl             # Trained LightGBM model
│   └── model_results.json         # Model performance metrics
│
├── artifacts/
│   ├── imputation_values.json     # Imputation statistics
│   ├── label_encoders.json        # Label encoding mappings
│   └── plots/                     # Model comparison plots
│
├── params.yaml                    # Configuration parameters
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup
└── README.md                      # Project documentation
```

## 🌐 Deployment

### Local Deployment

Already running! Just execute:
```bash
streamlit run app/streamlit_app.py
```

### Streamlit Cloud (Free)

1. Push code to GitHub
2. Visit [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your repository
4. Deploy with one click

### Docker Deployment

```bash
# Build image
docker build -t churn-predictor .

# Run container
docker run -p 8501:8501 churn-predictor
```

### Heroku Deployment

```bash
# Create Heroku app
heroku create your-app-name

# Push to Heroku
git push heroku main
```

See [STREAMLIT_DEPLOYMENT.md](STREAMLIT_DEPLOYMENT.md) for detailed deployment instructions.

## 📊 Key Insights

### Top Churn Indicators

1. **Tenure**: Early-stage customers (< 6 months) are high-risk
2. **Complaints**: Strong predictor of churn
3. **Order Frequency**: Low engagement indicates churn risk
4. **Satisfaction Score**: Directly correlates with retention
5. **Payment Mode**: Certain payment methods show higher churn

### Business Recommendations

- **Proactive Outreach**: Contact top 20% highest-risk customers
- **Retention Incentives**: Offer personalized discounts to at-risk customers
- **Improve Satisfaction**: Address complaint resolution processes
- **Engagement Programs**: Increase app usage and order frequency
- **Payment Flexibility**: Optimize payment options for retention

## 🛠️ Technologies Used

- **Python 3.8+**: Core programming language
- **Pandas & NumPy**: Data manipulation
- **Scikit-learn**: Machine learning utilities
- **LightGBM**: Gradient boosting framework
- **XGBoost**: Extreme gradient boosting
- **CatBoost**: Categorical boosting
- **SHAP**: Model explainability
- **Streamlit**: Web application framework
- **Matplotlib & Seaborn**: Data visualization
- **Kaggle API**: Data acquisition

## 📈 Future Enhancements

- [ ] Real-time prediction API with FastAPI
- [ ] A/B testing framework for retention strategies
- [ ] Customer lifetime value (CLV) prediction
- [ ] Automated model retraining pipeline
- [ ] Integration with CRM systems
- [ ] Advanced feature engineering (RFM analysis, customer segmentation)
- [ ] Multi-model ensemble for improved performance
- [ ] Deployment on cloud platforms (AWS, GCP, Azure)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Your Name**
- GitHub: [@rkpcode](https://github.com/rkpcode)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

## 🙏 Acknowledgments

- Dataset: [E-commerce Customer Churn Analysis and Prediction](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)
- SHAP Library: [SHAP Documentation](https://shap.readthedocs.io/)
- Streamlit: [Streamlit Documentation](https://docs.streamlit.io/)

## 📞 Support

For support, email your.email@example.com or open an issue in the GitHub repository.

---

⭐ **Star this repository if you find it helpful!**

Made with ❤️ for better customer retention