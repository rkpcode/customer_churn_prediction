# Streamlit Deployment Guide

## 🚀 Quick Start

The Streamlit app is now running at:
- **Local URL**: http://localhost:8501
- **Network URL**: http://10.39.37.179:8501

## 📱 App Features

### 1. Single Prediction Tab
- Interactive form for individual customer churn prediction
- Real-time risk assessment with color-coded results
- Actionable recommendations based on prediction
- Model confidence scores

### 2. Batch Prediction Tab
- Upload CSV files for bulk predictions
- Download sample CSV template
- Process multiple customers at once
- Export results with churn probabilities

### 3. Model Insights Tab
- Model performance comparison
- Business impact metrics
- Key insights and recommendations

## 🎯 How to Use

### Single Prediction
1. Navigate to "Single Prediction" tab
2. Fill in customer details:
   - Demographics (tenure, gender, marital status, city tier)
   - Purchase behavior (orders, last order date, cashback)
   - Engagement metrics (app usage, devices, satisfaction)
3. Click "Predict Churn Risk"
4. View results with risk level and recommendations

### Batch Prediction
1. Navigate to "Batch Prediction" tab
2. Download sample CSV template
3. Prepare your customer data in the same format
4. Upload CSV file
5. Click "Run Batch Prediction"
6. Download results with predictions

## 🛠️ Running the App

### Start the App
```bash
cd c:\DataScience_AI_folder\Portfolio\ecommerce_customer_churn
streamlit run app\streamlit_app.py
```

### Stop the App
Press `Ctrl+C` in the terminal

## 📦 Requirements

Make sure these packages are installed:
```bash
pip install streamlit pandas numpy scikit-learn lightgbm
```

## 🔧 Configuration

The app automatically loads:
- ✅ Trained LightGBM model from `models/best_model.pkl`
- ✅ Imputation values from `artifacts/imputation_values.json`
- ✅ Label encoders from `artifacts/label_encoders.json`
- ✅ Model results from `models/model_results.json`

## 📊 Model Performance

**Best Model**: LightGBM
- ROC-AUC: 0.9991
- Recall: 98.42%
- Precision: 92.57%
- F1-Score: 0.9541

**Threshold**: 0.247 (tuned for top 20% highest-risk customers)

## 🌐 Deployment Options

### Local Development
Already running! Access at http://localhost:8501

### Streamlit Cloud (Free)
1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Connect your GitHub repository
4. Deploy with one click

### Docker Deployment
Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t churn-predictor .
docker run -p 8501:8501 churn-predictor
```

### Heroku Deployment
1. Create `setup.sh`:
```bash
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

2. Create `Procfile`:
```
web: sh setup.sh && streamlit run app/streamlit_app.py
```

3. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

## 🎨 Customization

### Change Theme
Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

### Modify Threshold
Edit the threshold value in `streamlit_app.py`:
```python
threshold = model_results.get('threshold', 0.247)  # Change 0.247 to your value
```

## 🐛 Troubleshooting

### Model Not Loading
- Ensure `models/best_model.pkl` exists
- Run the training pipeline first if needed

### Import Errors
- Install missing packages: `pip install -r requirements.txt`
- Check Python version (3.8+ required)

### Port Already in Use
```bash
streamlit run app/streamlit_app.py --server.port=8502
```

## 📝 Next Steps

1. ✅ App is running locally
2. → Test single predictions
3. → Test batch predictions
4. → Deploy to Streamlit Cloud (optional)
5. → Share with stakeholders

## 🔗 Useful Links

- Streamlit Documentation: https://docs.streamlit.io
- Streamlit Cloud: https://streamlit.io/cloud
- GitHub Repository: (Add your repo link)

---

**Status**: ✅ Deployment Ready
**Access**: http://localhost:8501
