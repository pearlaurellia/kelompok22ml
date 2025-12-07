import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Bank Marketing Predictor",
    page_icon="🏦",
    layout="wide"
)

# Title and description
st.title("🏦 Bank Marketing Subscription Predictor")
st.markdown("Predict whether a customer will subscribe to a term deposit based on their profile and campaign data.")

# Load the trained model
@st.cache_resource
def load_model(model_path="outputs/pipeline_RandomForest.joblib"):
    """Load the trained model pipeline"""
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}. Please train the model first using tes_full.py")
        return None

# Model selection
st.sidebar.header("Model Selection")
model_options = {
    "Random Forest": "outputs/pipeline_RandomForest.joblib",
    "Random Forest (GridSearch)": "outputs/RandomForest_GridSearch.joblib",
    "Logistic Regression": "outputs/pipeline_LogisticRegression.joblib",
    "Decision Tree": "outputs/pipeline_DecisionTree.joblib",
    "XGBoost": "outputs/pipeline_XGBoost.joblib"
}

selected_model = st.sidebar.selectbox("Choose a model:", list(model_options.keys()))
model_path = model_options[selected_model]

# Load model
model = load_model(model_path)

if model is not None:
    st.sidebar.success(f"✅ {selected_model} loaded successfully!")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        job = st.selectbox("Job", [
            "admin.", "blue-collar", "entrepreneur", "housemaid", 
            "management", "retired", "self-employed", "services", 
            "student", "technician", "unemployed", "unknown"
        ])
        marital = st.selectbox("Marital Status", ["married", "single", "divorced", "unknown"])
        education = st.selectbox("Education", [
            "basic.4y", "basic.6y", "basic.9y", "high.school",
            "illiterate", "professional.course", "university.degree", "unknown"
        ])
        default = st.selectbox("Has Credit in Default?", ["no", "yes", "unknown"])
        housing = st.selectbox("Has Housing Loan?", ["no", "yes", "unknown"])
        loan = st.selectbox("Has Personal Loan?", ["no", "yes", "unknown"])
        
    with col2:
        st.subheader("Campaign Information")
        contact = st.selectbox("Contact Communication Type", ["cellular", "telephone"])
        month = st.selectbox("Last Contact Month", [
            "jan", "feb", "mar", "apr", "may", "jun",
            "jul", "aug", "sep", "oct", "nov", "dec"
        ])
        day_of_week = st.selectbox("Last Contact Day of Week", ["mon", "tue", "wed", "thu", "fri"])
        duration = st.number_input("Last Contact Duration (seconds)", min_value=0, max_value=5000, value=180)
        campaign = st.number_input("Number of Contacts This Campaign", min_value=1, max_value=50, value=2)
        pdays = st.number_input("Days Since Last Contact from Previous Campaign (999 = not contacted)", 
                                min_value=0, max_value=999, value=999)
        previous = st.number_input("Number of Contacts Before This Campaign", min_value=0, max_value=10, value=0)
        poutcome = st.selectbox("Outcome of Previous Campaign", ["nonexistent", "failure", "success"])
    
    # Economic indicators (usually fixed values)
    with st.expander("📊 Economic Indicators (Optional - Advanced)"):
        emp_var_rate = st.number_input("Employment Variation Rate", value=1.1, format="%.2f")
        cons_price_idx = st.number_input("Consumer Price Index", value=93.994, format="%.3f")
        cons_conf_idx = st.number_input("Consumer Confidence Index", value=-36.4, format="%.1f")
        euribor3m = st.number_input("Euribor 3 Month Rate", value=4.857, format="%.3f")
        nr_employed = st.number_input("Number of Employees (thousands)", value=5191.0, format="%.1f")
    
    # Predict button
    if st.button("🔮 Predict Subscription", type="primary"):
        # Create input dataframe matching training data structure
        input_data = pd.DataFrame({
            'age': [age],
            'job': [job],
            'marital': [marital],
            'education': [education],
            'default': [default],
            'housing': [housing],
            'loan': [loan],
            'contact': [contact],
            'month': [month],
            'day_of_week': [day_of_week],
            'duration': [duration],
            'campaign': [campaign],
            'pdays': [pdays],
            'previous': [previous],
            'poutcome': [poutcome],
            'emp.var.rate': [emp_var_rate],
            'cons.price.idx': [cons_price_idx],
            'cons.conf.idx': [cons_conf_idx],
            'euribor3m': [euribor3m],
            'nr.employed': [nr_employed]
        })
        
        # Make prediction
        try:
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            
            # Display results
            st.markdown("---")
            st.subheader("Prediction Results")
            
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                if prediction == 1:
                    st.success("### ✅ YES - Customer likely to subscribe!")
                    st.metric("Subscription Probability", f"{prediction_proba[1]*100:.1f}%")
                else:
                    st.error("### ❌ NO - Customer unlikely to subscribe")
                    st.metric("Non-Subscription Probability", f"{prediction_proba[0]*100:.1f}%")
            
            with result_col2:
                st.markdown("#### Confidence Breakdown")
                st.progress(float(prediction_proba[1]), text=f"Subscribe: {prediction_proba[1]*100:.1f}%")
                st.progress(float(prediction_proba[0]), text=f"Not Subscribe: {prediction_proba[0]*100:.1f}%")
            
            # Additional insights
            st.info(f"""
            **Interpretation:**
            - Contact duration is a strong predictor (longer calls often indicate interest)
            - Economic indicators affect subscription rates
            - Previous campaign outcomes matter significantly
            """)
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.error("Make sure the input features match the training data structure.")
    
    # Batch prediction option
    st.markdown("---")
    st.subheader("📁 Batch Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file for batch predictions", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, sep=';')
            st.write("Preview of uploaded data:", df.head())
            
            if st.button("Run Batch Predictions"):
                # Make predictions
                predictions = model.predict(df)
                probabilities = model.predict_proba(df)
                
                # Add results to dataframe
                df['prediction'] = predictions
                df['prediction_label'] = df['prediction'].map({0: 'no', 1: 'yes'})
                df['probability_yes'] = probabilities[:, 1]
                df['probability_no'] = probabilities[:, 0]
                
                st.success(f"✅ Predictions complete! {predictions.sum()} out of {len(predictions)} predicted to subscribe.")
                st.dataframe(df)
                
                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error processing file: {e}")

else:
    st.warning("⚠️ Please train a model first by running: `python tes_full.py`")
    st.info("""
    **Steps to get started:**
    1. Run `python tes_full.py --data bank-additional-full.csv` to train models
    2. Models will be saved in the `outputs/` folder
    3. Refresh this page to load the trained models
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Model trained using scikit-learn and imbalanced-learn")