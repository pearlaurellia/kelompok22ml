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

# Custom CSS with your color palette
st.markdown("""
<style>
    /* Main color palette:
       Primary Dark: #2D3142
       Gray: #BFC0C0
       White: #FFFFFF
       Orange: #EF8354
       Blue Gray: #4F5D75
    */
    
    /* Main background */
    .stApp {
        background: #FFFFFF;
    }
    
    /* Main content area */
    .main .block-container {
        padding: 2rem 3rem;
        background: #FFFFFF;
        max-width: 1400px;
    }
    
    /* Title styling */
    h1 {
        color: #2D3142 !important;
        font-weight: 800 !important;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 4px solid #EF8354;
        margin-bottom: 1rem !important;
    }
    
    h2, h3 {
        color: #4F5D75 !important;
        font-weight: 700 !important;
    }
    
    /* Subheader styling */
    .stMarkdown h2 {
        background: linear-gradient(90deg, #EF8354, #4F5D75);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Description text */
    .stMarkdown p {
        color: #4F5D75;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Model selection card */
    .model-card {
        background: linear-gradient(135deg, #2D3142 0%, #4F5D75 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(45, 49, 66, 0.2);
    }
    
    .model-card h3 {
        color: #EF8354 !important;
        margin-bottom: 1rem;
    }
    
    .model-card label {
        color: #FFFFFF !important;
    }
    
    /* Input fields */
    .stSelectbox, .stNumberInput {
        background-color: white;
        border-radius: 10px;
    }
    
    .stSelectbox > div > div,
    .stNumberInput > div > div {
        background-color: white;
        border: 2px solid #BFC0C0;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:hover,
    .stNumberInput > div > div:hover {
        border-color: #EF8354;
        box-shadow: 0 0 10px rgba(239, 131, 84, 0.3);
    }
    
    /* Labels */
    label {
        color: #2D3142 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #EF8354 0%, #ff6b35 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(239, 131, 84, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #ff6b35 0%, #EF8354 100%);
        box-shadow: 0 6px 20px rgba(239, 131, 84, 0.6);
        transform: translateY(-2px);
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #4F5D75 0%, #2D3142 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(79, 93, 117, 0.4);
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #2D3142 0%, #4F5D75 100%);
        transform: translateY(-2px);
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: rgba(76, 175, 80, 0.1);
        border-left: 4px solid #4CAF50;
        border-radius: 8px;
        padding: 1rem;
    }
    
    .stError {
        background-color: rgba(239, 131, 84, 0.1);
        border-left: 4px solid #EF8354;
        border-radius: 8px;
        padding: 1rem;
    }
    
    .stInfo {
        background-color: rgba(79, 93, 117, 0.1);
        border-left: 4px solid #4F5D75;
        border-radius: 8px;
        padding: 1rem;
    }
    
    .stWarning {
        background-color: rgba(191, 192, 192, 0.2);
        border-left: 4px solid #BFC0C0;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #EF8354 !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #4F5D75 !important;
        font-weight: 600 !important;
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #EF8354, #ff6b35);
        border-radius: 10px;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: rgba(79, 93, 117, 0.1);
        border-radius: 10px;
        border: 2px solid #BFC0C0;
        color: #2D3142 !important;
        font-weight: 600;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: rgba(239, 131, 84, 0.1);
        border-color: #EF8354;
    }
    
    /* Dataframe */
    .stDataFrame {
        border: 2px solid #BFC0C0;
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: rgba(79, 93, 117, 0.05);
        border: 2px dashed #BFC0C0;
        border-radius: 12px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #EF8354;
        background-color: rgba(239, 131, 84, 0.05);
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #BFC0C0, transparent);
        margin: 2rem 0;
    }
    
    /* Column containers */
    [data-testid="column"] {
        background-color: rgba(255, 255, 255, 0.6);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #BFC0C0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #4F5D75;
        padding: 1rem;
        margin-top: 2rem;
        border-top: 2px solid #BFC0C0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

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

# Model selection at the top
st.markdown("""
<div style="background: linear-gradient(135deg, #2D3142 0%, #4F5D75 100%); 
            padding: 2rem; border-radius: 15px; margin-bottom: 2rem; 
            box-shadow: 0 4px 15px rgba(45, 49, 66, 0.2);">
    <h3 style="color: #FFFFFF !important; margin-bottom: 1rem; margin-top: 0;">🤖 Model Selection</h3>
</div>
""", unsafe_allow_html=True)

model_col1, model_col2 = st.columns([3, 1])

with model_col1:
    model_options = {
        "Random Forest": "outputs/pipeline_RandomForest.joblib",
        "Random Forest (GridSearch)": "outputs/RandomForest_GridSearch.joblib",
        "Logistic Regression": "outputs/pipeline_LogisticRegression.joblib",
        "Decision Tree": "outputs/pipeline_DecisionTree.joblib",
        "XGBoost": "outputs/pipeline_XGBoost.joblib"
    }
    
    selected_model = st.selectbox("Choose a model:", list(model_options.keys()))
    model_path = model_options[selected_model]

with model_col2:
    st.markdown("<br>", unsafe_allow_html=True)
    # Load model
    model = load_model(model_path)

if model is not None:
    st.success(f"✅ {selected_model} loaded successfully!")
    
    st.markdown("---")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Personal Information")
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
        st.subheader("📞 Campaign Information")
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
    
    st.markdown("<br>", unsafe_allow_html=True)
    
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
            st.subheader("🎯 Prediction Results")
            
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                if prediction == 1:
                    st.success("### ✅ YES - Customer likely to subscribe!")
                    st.metric("Subscription Probability", f"{prediction_proba[1]*100:.1f}%")
                else:
                    st.error("### ❌ NO - Customer unlikely to subscribe")
                    st.metric("Non-Subscription Probability", f"{prediction_proba[0]*100:.1f}%")
            
            with result_col2:
                st.markdown("#### 📊 Confidence Breakdown")
                st.progress(float(prediction_proba[1]), text=f"Subscribe: {prediction_proba[1]*100:.1f}%")
                st.progress(float(prediction_proba[0]), text=f"Not Subscribe: {prediction_proba[0]*100:.1f}%")
            
            # Additional insights
            st.info(f"""
            **💡 Interpretation:**
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
            st.write("**Preview of uploaded data:**")
            st.dataframe(df.head())
            
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
st.markdown(
    '<div class="footer">Built with ❤️ using Streamlit | Model trained using scikit-learn and imbalanced-learn</div>', 
    unsafe_allow_html=True
)
