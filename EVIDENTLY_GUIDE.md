# Evidently AI - Model Monitoring Setup

## 🎯 What is Evidently AI?

Evidently AI is a production ML monitoring tool that helps you:
- **Detect Data Drift**: Know when your production data changes
- **Monitor Model Performance**: Track accuracy degradation over time
- **Ensure Data Quality**: Catch data issues before they affect predictions
- **Generate Reports**: Beautiful HTML reports for stakeholders

## ✅ Setup Complete

### What We've Configured:

1. **Model Monitoring Component** ✅
   - `src/ecommerce_customer_churn/components/model_monitoring.py`
   - Data drift detection
   - Data quality monitoring
   - Model performance tracking

2. **Report Types** ✅
   - **Data Drift Report**: Detects distribution changes
   - **Data Quality Report**: Checks for missing values, outliers
   - **Model Performance Report**: Tracks accuracy, precision, recall

## 🚀 Usage

### 1. Basic Monitoring

```python
from src.ecommerce_customer_churn.components.model_monitoring import ModelMonitoring
from pathlib import Path

# Initialize monitoring
monitoring = ModelMonitoring()

# Monitor production data
results = monitoring.monitor_production_data(
    reference_data_path=Path("data/processed/X_train_phase2.csv"),
    current_data_path=Path("data/new_production_data.csv"),
    target_col='Churn'
)
```

### 2. Data Drift Detection Only

```python
import pandas as pd

# Load data
reference_data = pd.read_csv("data/processed/X_train_phase2.csv")
current_data = pd.read_csv("data/new_data.csv")

# Check for drift
drift_summary = monitoring.generate_data_drift_report(
    reference_data=reference_data,
    current_data=current_data,
    target_col='Churn'
)

if drift_summary['dataset_drift_detected']:
    print("⚠️ DATA DRIFT DETECTED!")
    print(f"Drift Share: {drift_summary['drift_share']:.2%}")
else:
    print("✅ No significant drift")
```

### 3. Model Performance Monitoring

```python
# Add predictions to your data
current_data['prediction'] = model.predict(current_data)

# Monitor performance
performance_summary = monitoring.generate_model_performance_report(
    reference_data=reference_data_with_predictions,
    current_data=current_data,
    target_col='Churn',
    prediction_col='prediction'
)
```

## 📊 Generated Reports

All reports are saved in `reports/` directory:

1. **`data_drift_report.html`**
   - Feature-by-feature drift analysis
   - Statistical tests (KS test, Chi-squared)
   - Visual comparisons

2. **`data_quality_report.html`**
   - Missing values analysis
   - Outlier detection
   - Data type consistency

3. **`model_performance_report.html`**
   - Confusion matrix
   - Classification metrics
   - Performance by class

4. **`monitoring_summary.json`**
   - Machine-readable summary
   - Drift flags
   - Timestamps

## 🔔 Setting Up Alerts

### Option 1: Email Alerts

```python
import smtplib
from email.mime.text import MIMEText

def send_drift_alert(drift_summary):
    if drift_summary['dataset_drift_detected']:
        msg = MIMEText(f"Data drift detected! Drift share: {drift_summary['drift_share']:.2%}")
        msg['Subject'] = '⚠️ ML Model Alert: Data Drift Detected'
        msg['From'] = 'alerts@yourcompany.com'
        msg['To'] = 'ml-team@yourcompany.com'
        
        # Send email
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login('your_email', 'your_password')
            server.send_message(msg)
```

### Option 2: Slack Alerts

```python
import requests

def send_slack_alert(drift_summary):
    if drift_summary['dataset_drift_detected']:
        webhook_url = "YOUR_SLACK_WEBHOOK_URL"
        message = {
            "text": f"⚠️ Data Drift Detected!\nDrift Share: {drift_summary['drift_share']:.2%}\nReport: {drift_summary['report_path']}"
        }
        requests.post(webhook_url, json=message)
```

## 🔄 Production Workflow

### Daily Monitoring Script

```python
# monitor_daily.py
from src.ecommerce_customer_churn.components.model_monitoring import ModelMonitoring
from pathlib import Path
import pandas as pd
from datetime import datetime

def daily_monitoring():
    monitoring = ModelMonitoring()
    
    # Load reference data (training data)
    reference_data = pd.read_csv("data/processed/X_train_phase2.csv")
    
    # Load today's production data
    today = datetime.now().strftime("%Y-%m-%d")
    current_data = pd.read_csv(f"data/production/{today}_predictions.csv")
    
    # Run monitoring
    results = monitoring.monitor_production_data(
        reference_data_path=Path("data/processed/X_train_phase2.csv"),
        current_data_path=Path(f"data/production/{today}_predictions.csv"),
        target_col='Churn',
        prediction_col='prediction'
    )
    
    # Send alerts if needed
    if results['data_drift']['dataset_drift_detected']:
        send_drift_alert(results['data_drift'])
    
    return results

if __name__ == "__main__":
    daily_monitoring()
```

### Schedule with Cron (Linux/Mac)

```bash
# Run monitoring every day at 2 AM
0 2 * * * cd /path/to/project && python monitor_daily.py
```

### Schedule with Task Scheduler (Windows)

```powershell
# Create scheduled task
$action = New-ScheduledTaskAction -Execute "python" -Argument "monitor_daily.py" -WorkingDirectory "C:\path\to\project"
$trigger = New-ScheduledTaskTrigger -Daily -At 2am
Register-ScheduledTask -Action $action -Trigger $trigger -TaskName "ML_Monitoring"
```

## 📈 Interpreting Results

### Data Drift Thresholds

- **< 10% drift**: Normal variation, no action needed
- **10-30% drift**: Monitor closely, investigate features
- **> 30% drift**: ⚠️ Retrain model, update features

### When to Retrain

Retrain your model if:
1. Data drift > 30%
2. Model performance drops > 5%
3. New patterns emerge in production
4. Business requirements change

## 🛠️ Integration with Streamlit

Add monitoring tab to your Streamlit app:

```python
# In streamlit_app.py
with tab_monitoring:
    st.header("📊 Model Monitoring")
    
    if st.button("Generate Monitoring Reports"):
        monitoring = ModelMonitoring()
        results = monitoring.monitor_production_data(...)
        
        st.success("Reports generated!")
        
        # Display drift status
        if results['data_drift']['dataset_drift_detected']:
            st.error(f"⚠️ Data Drift Detected: {results['data_drift']['drift_share']:.2%}")
        else:
            st.success("✅ No significant drift")
        
        # Embed HTML reports
        with open("reports/data_drift_report.html", "r") as f:
            st.components.v1.html(f.read(), height=800, scrolling=True)
```

## 📝 Best Practices

1. **Monitor Regularly**: Daily or weekly depending on data volume
2. **Set Baselines**: Use stable training data as reference
3. **Track Trends**: Look for gradual drift over time
4. **Document Actions**: Keep log of retraining decisions
5. **Automate Alerts**: Don't rely on manual checks

## 🔗 Useful Links

- [Evidently Documentation](https://docs.evidentlyai.com/)
- [Evidently GitHub](https://github.com/evidentlyai/evidently)
- [Example Reports](https://docs.evidentlyai.com/examples)

---

**Status**: ✅ Evidently AI Configured and Ready for Production Monitoring!
