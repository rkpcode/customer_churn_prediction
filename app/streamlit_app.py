"""
Streamlit App for E-commerce Customer Churn Prediction
Production-ready deployment with trained LightGBM model + SHAP Explainability
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
import os
import shap
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ecommerce_customer_churn.utils import load_object

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .high-risk {
        background-color: #ffebee;
        border-left-color: #f44336;
    }
    .low-risk {
        background-color: #e8f5e9;
        border-left-color: #4caf50;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_artifacts():
    """Load trained model and preprocessing artifacts"""
    try:
        # Load model
        model_path = project_root / "models" / "best_model.pkl"
        model = load_object(model_path)
        
        # Load imputation values
        imputation_path = project_root / "artifacts" / "imputation_values.json"
        with open(imputation_path, 'r') as f:
            imputation_values = json.load(f)
        
        # Load label encoders
        encoders_path = project_root / "artifacts" / "label_encoders.json"
        with open(encoders_path, 'r') as f:
            label_encoders = json.load(f)
        
        # Load model results
        results_path = project_root / "models" / "model_results.json"
        with open(results_path, 'r') as f:
            model_results = json.load(f)
        
        # Load training data for SHAP background
        X_train_path = project_root / "data" / "processed" / "X_train_phase2.csv"
        X_train = pd.read_csv(X_train_path)
        
        # Create SHAP explainer (use sample for speed)
        explainer = shap.TreeExplainer(model, X_train.sample(min(100, len(X_train)), random_state=42))
        
        return model, imputation_values, label_encoders, model_results, explainer, X_train
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None, None, None


def get_feature_names():
    """Get feature names in correct order"""
    return [
        'Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp',
        'NumberOfDeviceRegistered', 'SatisfactionScore', 'NumberOfAddress',
        'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount',
        'DaySinceLastOrder', 'CashbackAmount', 'PreferredLoginDevice',
        'PreferredPaymentMode', 'Gender', 'PreferedOrderCat', 'MaritalStatus',
        'Tenure_was_missing', 'HourSpendOnApp_was_missing', 'OrderCount_was_missing',
        'DaySinceLastOrder_was_missing', 'OrderAmountHikeFromlastYear_was_missing',
        'CouponUsed_was_missing', 'order_frequency', 'complaint_rate'
    ]


def plot_shap_waterfall(explainer, processed_data):
    """Create SHAP waterfall plot for single prediction"""
    try:
        shap_values = explainer(processed_data)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.plots.waterfall(shap_values[0], max_display=15, show=False)
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error creating SHAP waterfall plot: {e}")
        return None


def get_feature_importance(explainer, X_train):
    """Get global feature importance"""
    try:
        shap_values = explainer(X_train.sample(min(100, len(X_train)), random_state=42))
        
        # Calculate mean absolute SHAP values
        feature_importance = pd.DataFrame({
            'feature': get_feature_names(),
            'importance': np.abs(shap_values.values).mean(axis=0)
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    except Exception as e:
        st.error(f"Error calculating feature importance: {e}")
        return None


def preprocess_input(input_data, imputation_values, label_encoders):
    """Preprocess user input to match training data format"""
    
    # Create DataFrame
    df = pd.DataFrame([input_data])
    
    # Handle missing values (use training medians/modes)
    for col, info in imputation_values.items():
        if col in df.columns and pd.isna(df[col].iloc[0]):
            df[col] = info['value']
    
    # Create missing flags
    missing_flag_cols = ['Tenure', 'HourSpendOnApp', 'OrderCount', 
                         'DaySinceLastOrder', 'OrderAmountHikeFromlastYear', 'CouponUsed']
    for col in missing_flag_cols:
        if col in df.columns:
            df[f'{col}_was_missing'] = 0  # User input assumed complete
    
    # Label encode categorical features
    categorical_cols = ['PreferredLoginDevice', 'PreferredPaymentMode', 
                       'Gender', 'PreferedOrderCat', 'MaritalStatus']
    
    for col in categorical_cols:
        if col in df.columns and col in label_encoders:
            value = df[col].iloc[0]
            if value in label_encoders[col]:
                df[col] = label_encoders[col].index(value)
            else:
                df[col] = -1  # Unseen category
    
    # Add Phase 2 features
    df['order_frequency'] = df['OrderCount'] / (df['Tenure'] + 1)
    df['complaint_rate'] = df['Complain'] / (df['OrderCount'] + 1)
    
    # Ensure correct column order (26 features for Phase 2)
    expected_columns = get_feature_names()
    
    # Reorder columns
    df = df[expected_columns]
    
    return df


def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 Customer Churn Prediction with AI Explainability</h1>', unsafe_allow_html=True)
    
    # Load model and artifacts
    model, imputation_values, label_encoders, model_results, explainer, X_train = load_model_and_artifacts()
    
    if model is None:
        st.error("⚠️ Failed to load model. Please ensure the model is trained and saved.")
        return
    
    # Sidebar - Model Information
    with st.sidebar:
        st.header("📊 Model Information")
        
        if model_results:
            best_model_name = "LightGBM"
            if best_model_name in model_results:
                metrics = model_results[best_model_name]
                st.metric("Model", best_model_name)
                st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
                st.metric("Recall", f"{metrics.get('recall', 0)*100:.2f}%")
                st.metric("Precision", f"{metrics.get('precision', 0)*100:.2f}%")
                st.metric("F1-Score", f"{metrics.get('f1_score', 0):.4f}")
        
        st.markdown("---")
        st.markdown("### 🎓 About")
        st.info("""
        This model predicts customer churn risk with **SHAP explainability** to show:
        - Which features drive predictions
        - How each feature impacts the result
        - Global feature importance
        """)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Single Prediction", "📁 Batch Prediction", "📈 Model Insights", "🔍 Explainability"])
    
    with tab1:
        st.header("Single Customer Prediction")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📋 Customer Demographics")
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=10)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            city_tier = st.selectbox("City Tier", [1, 2, 3])
            
        with col2:
            st.subheader("🛒 Purchase Behavior")
            order_count = st.number_input("Order Count", min_value=0, max_value=50, value=5)
            days_since_last_order = st.number_input("Days Since Last Order", min_value=0, max_value=365, value=10)
            preferred_order_cat = st.selectbox("Preferred Order Category", 
                                              ["Laptop & Accessory", "Mobile Phone", "Fashion", 
                                               "Grocery", "Others", "Mobile"])
            cashback_amount = st.number_input("Cashback Amount", min_value=0.0, max_value=500.0, value=100.0)
            coupon_used = st.number_input("Coupons Used", min_value=0, max_value=20, value=2)
            order_hike = st.number_input("Order Amount Hike (%)", min_value=0, max_value=100, value=10)
        
        with col3:
            st.subheader("📱 Engagement & Satisfaction")
            hour_spend_on_app = st.number_input("Hours Spend on App", min_value=0.0, max_value=10.0, value=2.0)
            num_devices = st.number_input("Number of Devices Registered", min_value=1, max_value=10, value=2)
            preferred_login_device = st.selectbox("Preferred Login Device", 
                                                 ["Mobile Phone", "Computer", "Phone"])
            satisfaction_score = st.slider("Satisfaction Score", min_value=1, max_value=5, value=3)
            complain = st.selectbox("Has Complained?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            
            warehouse_to_home = st.number_input("Warehouse to Home Distance (km)", 
                                               min_value=0, max_value=100, value=15)
            num_address = st.number_input("Number of Addresses", min_value=1, max_value=10, value=2)
            preferred_payment = st.selectbox("Preferred Payment Mode", 
                                            ["Debit Card", "Credit Card", "E wallet", 
                                             "UPI", "COD", "Cash on Delivery", "CC"])
        
        # Predict button
        if st.button("🔮 Predict Churn Risk with Explanation", type="primary"):
            # Prepare input
            input_data = {
                'Tenure': tenure,
                'CityTier': city_tier,
                'WarehouseToHome': warehouse_to_home,
                'HourSpendOnApp': hour_spend_on_app,
                'NumberOfDeviceRegistered': num_devices,
                'SatisfactionScore': satisfaction_score,
                'NumberOfAddress': num_address,
                'Complain': complain,
                'OrderAmountHikeFromlastYear': order_hike,
                'CouponUsed': coupon_used,
                'OrderCount': order_count,
                'DaySinceLastOrder': days_since_last_order,
                'CashbackAmount': cashback_amount,
                'PreferredLoginDevice': preferred_login_device,
                'PreferredPaymentMode': preferred_payment,
                'Gender': gender,
                'PreferedOrderCat': preferred_order_cat,
                'MaritalStatus': marital_status
            }
            
            # Preprocess
            processed_data = preprocess_input(input_data, imputation_values, label_encoders)
            
            # Predict
            prediction_proba = model.predict_proba(processed_data)[0]
            churn_probability = prediction_proba[1]
            
            # Business threshold (from training)
            threshold = model_results.get('threshold', 0.247)
            prediction = 1 if churn_probability >= threshold else 0
            
            # Display results
            st.markdown("---")
            st.header("📊 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                risk_level = "HIGH RISK" if prediction == 1 else "LOW RISK"
                risk_color = "🔴" if prediction == 1 else "🟢"
                st.markdown(f"""
                <div class="metric-card {'high-risk' if prediction == 1 else 'low-risk'}">
                    <h2>{risk_color} {risk_level}</h2>
                    <p style="font-size: 1.2rem;">Churn Probability: <strong>{churn_probability*100:.2f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                confidence = max(prediction_proba)
                st.metric("Model Confidence", f"{confidence*100:.2f}%")
                st.metric("Decision Threshold", f"{threshold:.3f}")
            
            with col3:
                if prediction == 1:
                    st.error("⚠️ **Action Required**")
                    st.markdown("""
                    **Recommended Actions:**
                    - Offer personalized retention incentive
                    - Schedule customer success call
                    - Provide exclusive discount
                    - Improve customer experience
                    """)
                else:
                    st.success("✅ **Customer Stable**")
                    st.markdown("""
                    **Recommended Actions:**
                    - Continue regular engagement
                    - Upsell opportunities available
                    - Maintain service quality
                    """)
            
            # SHAP Explanation
            st.markdown("---")
            st.header("🔍 AI Explanation - Why this prediction?")
            
            with st.spinner("Generating SHAP explanation..."):
                shap_fig = plot_shap_waterfall(explainer, processed_data)
                if shap_fig:
                    st.pyplot(shap_fig)
                    st.caption("**Waterfall Plot**: Shows how each feature pushes the prediction from the base value (average prediction) towards the final prediction. Red bars push towards churn, blue bars push away from churn.")
    
    with tab2:
        st.header("Batch Prediction")
        st.info("📁 Upload a CSV file with customer data for batch predictions")
        
        # Sample CSV download
        sample_data = {
            'Tenure': [10, 5, 20],
            'CityTier': [1, 2, 3],
            'WarehouseToHome': [15, 20, 10],
            'HourSpendOnApp': [2.0, 1.5, 3.0],
            'NumberOfDeviceRegistered': [2, 1, 3],
            'SatisfactionScore': [3, 2, 4],
            'NumberOfAddress': [2, 1, 3],
            'Complain': [0, 1, 0],
            'OrderAmountHikeFromlastYear': [10, 5, 15],
            'CouponUsed': [2, 1, 3],
            'OrderCount': [5, 3, 8],
            'DaySinceLastOrder': [10, 20, 5],
            'CashbackAmount': [100, 50, 150],
            'PreferredLoginDevice': ['Mobile Phone', 'Computer', 'Mobile Phone'],
            'PreferredPaymentMode': ['Debit Card', 'Credit Card', 'UPI'],
            'Gender': ['Male', 'Female', 'Male'],
            'PreferedOrderCat': ['Mobile Phone', 'Fashion', 'Laptop & Accessory'],
            'MaritalStatus': ['Single', 'Married', 'Single']
        }
        sample_df = pd.DataFrame(sample_data)
        
        st.download_button(
            label="📥 Download Sample CSV",
            data=sample_df.to_csv(index=False),
            file_name="sample_customer_data.csv",
            mime="text/csv"
        )
        
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(batch_df)} customers")
                
                if st.button("🚀 Run Batch Prediction"):
                    predictions = []
                    probabilities = []
                    
                    progress_bar = st.progress(0)
                    for idx, row in batch_df.iterrows():
                        processed = preprocess_input(row.to_dict(), imputation_values, label_encoders)
                        proba = model.predict_proba(processed)[0][1]
                        pred = 1 if proba >= model_results.get('threshold', 0.247) else 0
                        
                        predictions.append(pred)
                        probabilities.append(proba)
                        progress_bar.progress((idx + 1) / len(batch_df))
                    
                    # Add results to dataframe
                    batch_df['Churn_Probability'] = probabilities
                    batch_df['Churn_Prediction'] = predictions
                    batch_df['Risk_Level'] = batch_df['Churn_Prediction'].map({0: 'Low Risk', 1: 'High Risk'})
                    
                    # Display results
                    st.header("📊 Batch Prediction Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Customers", len(batch_df))
                    with col2:
                        high_risk = (batch_df['Churn_Prediction'] == 1).sum()
                        st.metric("High Risk Customers", high_risk, 
                                 delta=f"{high_risk/len(batch_df)*100:.1f}%")
                    with col3:
                        avg_prob = batch_df['Churn_Probability'].mean()
                        st.metric("Average Churn Probability", f"{avg_prob*100:.2f}%")
                    
                    # Show results table
                    st.dataframe(batch_df.sort_values('Churn_Probability', ascending=False))
                    
                    # Download results
                    st.download_button(
                        label="📥 Download Predictions",
                        data=batch_df.to_csv(index=False),
                        file_name="churn_predictions.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    with tab3:
        st.header("Model Performance Insights")
        
        if model_results:
            st.subheader("📈 Model Comparison")
            
            # Create comparison dataframe
            models_to_show = ['Dumb Baseline', 'Logistic Regression', 'Random Forest', 
                             'Gradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost']
            
            comparison_data = []
            for model_name in models_to_show:
                if model_name in model_results:
                    metrics = model_results[model_name]
                    comparison_data.append({
                        'Model': model_name,
                        'Accuracy': f"{metrics.get('accuracy', 0)*100:.2f}%",
                        'Precision': f"{metrics.get('precision', 0)*100:.2f}%",
                        'Recall': f"{metrics.get('recall', 0)*100:.2f}%",
                        'F1-Score': f"{metrics.get('f1_score', 0):.4f}",
                        'ROC-AUC': f"{metrics.get('roc_auc', 0):.4f}"
                    })
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, width=1200)
            
            st.markdown("---")
            st.subheader("🎯 Business Impact")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                ### ✅ Model Advantages
                - **99.91% ROC-AUC**: Excellent discrimination
                - **98.42% Recall**: Catches almost all churners
                - **92.57% Precision**: High efficiency
                - **Threshold Tuning**: Business-aligned decisions
                """)
            
            with col2:
                st.markdown("""
                ### 💡 Key Insights
                - Early-stage customers are high-risk
                - Complaints strongly indicate churn
                - Payment mode affects retention
                - Engagement metrics are crucial
                """)
    
    with tab4:
        st.header("🔍 Model Explainability (SHAP)")
        
        st.markdown("""
        ### What is SHAP?
        SHAP (SHapley Additive exPlanations) is a game-theoretic approach to explain machine learning predictions.
        It shows **how much each feature contributes** to pushing the prediction away from the average.
        """)
        
        st.markdown("---")
        st.subheader("📊 Global Feature Importance")
        
        with st.spinner("Calculating feature importance..."):
            feature_importance = get_feature_importance(explainer, X_train)
            
            if feature_importance is not None:
                # Plot top 15 features
                top_features = feature_importance.head(15)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(top_features['feature'], top_features['importance'], color='steelblue')
                ax.set_xlabel('Mean |SHAP Value| (Average Impact on Prediction)', fontsize=12)
                ax.set_ylabel('Feature', fontsize=12)
                ax.set_title('Top 15 Most Important Features', fontsize=14, fontweight='bold')
                ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)
                
                st.caption("**Feature Importance**: Shows which features have the biggest impact on predictions across all customers.")
                
                # Show table
                st.dataframe(feature_importance.head(15), use_container_width=True)
        
        st.markdown("---")
        st.info("""
        💡 **How to use this information:**
        - Focus retention efforts on customers with high-risk feature values
        - Improve features that strongly predict churn (e.g., satisfaction, complaints)
        - Monitor key indicators like tenure and order frequency
        """)


if __name__ == "__main__":
    main()
