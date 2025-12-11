import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Bank Marketing Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Proper UI/UX Hierarchy
st.markdown("""
    <style>
    /* Typography Scale (following 1.25 ratio - Major Third) */
    
    /* Sidebar - Compact but readable */
    [data-testid="stSidebar"] {
        font-size: 14px !important;
    }
    [data-testid="stSidebar"] * {
        font-size: 14px !important;
        line-height: 1.5 !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2 {
        font-size: 16px !important;
        font-weight: 700 !important;
        margin: 10px 0 8px 0 !important;
        color: #2c3e50 !important;
    }
    [data-testid="stSidebar"] h3 {
        font-size: 15px !important;
        font-weight: 600 !important;
        margin: 8px 0 6px 0 !important;
        color: #34495e !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        font-size: 24px !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        font-size: 13px !important;
        font-weight: 500 !important;
    }
    
    /* Main content - Clear hierarchy */
    .main {
        font-size: 16px !important;
    }
    
    .main h1 {
        font-size: 32px !important;
        font-weight: 700 !important;
        color: #1a1a1a !important;
        margin: 30px 0 15px 0 !important;
        line-height: 1.2 !important;
    }
    
    .main h2 {
        font-size: 26px !important;
        font-weight: 700 !important;
        color: #2c3e50 !important;
        margin: 25px 0 12px 0 !important;
        line-height: 1.3 !important;
    }
    
    .main h3 {
        font-size: 22px !important;
        font-weight: 700 !important;
        color: #2c3e50 !important;
        margin: 25px 0 15px 0 !important;
        line-height: 1.4 !important;
    }
    
    .main h4 {
        font-size: 18px !important;
        font-weight: 600 !important;
        color: #34495e !important;
        margin: 20px 0 12px 0 !important;
        line-height: 1.4 !important;
    }
    
    /* Body text */
    .main p, .main label {
        font-size: 16px !important;
        line-height: 1.6 !important;
        color: #4a5568 !important;
    }
    
    /* Tabs - Clear and accessible */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px !important;
        border-bottom: 2px solid #e2e8f0 !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 16px !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        color: #64748b !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #1f77b4 !important;
        border-bottom: 3px solid #1f77b4 !important;
    }
    
    /* Buttons - Accessible size */
    .stButton>button {
        background-color: #1f77b4 !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button:hover {
        background-color: #1557b0 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(31, 119, 180, 0.3) !important;
    }
    
    /* Input fields - Better spacing */
    .stTextInput input,
    .stNumberInput input,
    .stSelectbox select {
        font-size: 15px !important;
        padding: 10px 12px !important;
    }
    
    /* Metrics - Visual hierarchy */
    [data-testid="stMetric"] {
        background-color: #f8fafc !important;
        padding: 16px !important;
        border-radius: 8px !important;
        border-left: 4px solid #1f77b4 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 14px !important;
        font-weight: 500 !important;
        color: #64748b !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: 700 !important;
        color: #1a1a1a !important;
    }
    
    /* Dividers - Subtle separation */
    hr {
        margin: 30px 0 !important;
        border-color: #e2e8f0 !important;
    }
    
    /* Expander - Clear affordance */
    .streamlit-expanderHeader {
        font-size: 16px !important;
        font-weight: 600 !important;
        color: #2c3e50 !important;
    }
    
    /* Info boxes */
    .stAlert {
        font-size: 15px !important;
        line-height: 1.6 !important;
        padding: 16px !important;
        border-radius: 8px !important;
    }
    
    /* Captions - Subtle text */
    .caption {
        font-size: 13px !important;
        color: #718096 !important;
        line-height: 1.4 !important;
    }
    </style>
""", unsafe_allow_html=True)

# TITLE - Proper size with good hierarchy
st.markdown("""
<div style="text-align: center; margin: 50px 0 30px 0;">
    <h1 style="font-size: 64px; font-weight: 900; color: #1f77b4; margin: 0; line-height: 1.1; text-shadow: 3px 3px 8px rgba(0,0,0,0.15); letter-spacing: -1px;">
        🏦 Bank Marketing Predictor
    </h1>
    <p style="font-size: 18px; color: #64748b; margin-top: 20px; font-weight: 400; line-height: 1.6; max-width: 800px; margin-left: auto; margin-right: auto;">
        Predict customer subscription likelihood for term deposits using machine learning
    </p>
</div>
""", unsafe_allow_html=True)

# Load model function
@st.cache_resource
def load_model(model_path):
    try:
        return joblib.load(model_path)
    except:
        return None

# Helper dictionaries
JOB_LABELS = {
    "admin.": "Administrative", "blue-collar": "Blue-collar Worker",
    "entrepreneur": "Entrepreneur", "housemaid": "Housemaid",
    "management": "Management", "retired": "Retired",
    "self-employed": "Self-employed", "services": "Services",
    "student": "Student", "technician": "Technician",
    "unemployed": "Unemployed", "unknown": "Unknown"
}

EDUCATION_LABELS = {
    "basic.4y": "Basic Education (4 years)", "basic.6y": "Basic Education (6 years)",
    "basic.9y": "Basic Education (9 years)", "high.school": "High School",
    "illiterate": "Illiterate", "professional.course": "Professional Course",
    "university.degree": "University Degree", "unknown": "Unknown"
}

MONTH_LABELS = {
    "jan": "January", "feb": "February", "mar": "March", "apr": "April",
    "may": "May", "jun": "June", "jul": "July", "aug": "August",
    "sep": "September", "oct": "October", "nov": "November", "dec": "December"
}

DAY_LABELS = {
    "mon": "Monday", "tue": "Tuesday", "wed": "Wednesday",
    "thu": "Thursday", "fri": "Friday"
}

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-building.png", width=70)
    st.header("⚙️ Model Configuration")
    
    model_options = {
        "🌲 Random Forest": "outputs/pipeline_RandomForest.joblib",
        "🌲 Random Forest (Optimized)": "outputs/RandomForest_GridSearch.joblib",
        "📊 Logistic Regression": "outputs/pipeline_LogisticRegression.joblib",
        "🌳 Decision Tree": "outputs/pipeline_DecisionTree.joblib",
    }
    
    selected_model_display = st.selectbox("Select ML Model:", list(model_options.keys()))
    model = load_model(model_options[selected_model_display])
    
    if model:
        st.success("✅ Model loaded!")
        with st.expander("📋 Info"):
            st.markdown(f"**Model:** {selected_model_display}")
    else:
        st.error("❌ Model not found")
    
    st.divider()
    st.subheader("📊 Quick Stats")
    st.metric("Models Available", len([p for p in model_options.values() if Path(p).exists()]))
    st.metric("Last Updated", datetime.now().strftime("%Y-%m-%d"))

# Main content
if model:
    tab1, tab2, tab3 = st.tabs(["🎯 Single Prediction", "📁 Batch Prediction", "📈 Model Performance"])
    
    with tab1:
        st.markdown("### Customer Information")
        
        st.markdown("#### 👤 Demographics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", 18, 100, 35)
        with col2:
            job_display = st.selectbox("Occupation", list(JOB_LABELS.values()))
            job = [k for k, v in JOB_LABELS.items() if v == job_display][0]
        with col3:
            marital_display = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Unknown"])
            marital = marital_display.lower()
        
        col1, col2 = st.columns(2)
        with col1:
            education_display = st.selectbox("Education Level", list(EDUCATION_LABELS.values()))
            education = [k for k, v in EDUCATION_LABELS.items() if v == education_display][0]
        
        st.divider()
        
        st.markdown("#### 💳 Financial Profile")
        col1, col2, col3 = st.columns(3)
        with col1:
            default_display = st.selectbox("Credit in Default?", ["No", "Yes", "Unknown"])
            default = default_display.lower()
        with col2:
            housing_display = st.selectbox("Housing Loan?", ["No", "Yes", "Unknown"])
            housing = housing_display.lower()
        with col3:
            loan_display = st.selectbox("Personal Loan?", ["No", "Yes", "Unknown"])
            loan = loan_display.lower()
        
        st.divider()
        
        st.markdown("#### 📞 Campaign Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            contact_display = st.selectbox("Contact Method", ["Cellular", "Telephone"])
            contact = contact_display.lower()
        with col2:
            month_display = st.selectbox("Last Contact Month", list(MONTH_LABELS.values()))
            month = [k for k, v in MONTH_LABELS.items() if v == month_display][0]
        with col3:
            day_display = st.selectbox("Last Contact Day", list(DAY_LABELS.values()))
            day_of_week = [k for k, v in DAY_LABELS.items() if v == day_display][0]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            duration = st.number_input("Call Duration (seconds)", 0, 5000, 180)
            st.caption(f"≈ {duration // 60} min {duration % 60} sec")
        with col2:
            campaign = st.number_input("Contacts This Campaign", 1, 50, 2)
        with col3:
            previous = st.number_input("Previous Contacts", 0, 10, 0)
        
        col1, col2 = st.columns(2)
        with col1:
            pdays = st.number_input("Days Since Last Contact", 0, 999, 999)
            st.caption("⚠️ Not previously contacted" if pdays == 999 else f"≈ {pdays // 30} months ago")
        with col2:
            poutcome_display = st.selectbox("Previous Campaign Outcome", ["Non-existent", "Failure", "Success"])
            poutcome = poutcome_display.lower().replace("-", "")
        
        with st.expander("📊 Economic Indicators"):
            col1, col2 = st.columns(2)
            with col1:
                emp_var_rate = st.number_input("Employment Variation Rate", value=1.1, format="%.2f")
                cons_price_idx = st.number_input("Consumer Price Index", value=93.994, format="%.3f")
                cons_conf_idx = st.number_input("Consumer Confidence Index", value=-36.4, format="%.1f")
            with col2:
                euribor3m = st.number_input("Euribor 3 Month Rate", value=4.857, format="%.3f")
                nr_employed = st.number_input("Number of Employees (thousands)", value=5191.0, format="%.1f")
        
        st.divider()
        
        if st.button("🔮 Predict Subscription Likelihood", type="primary", use_container_width=True):
            input_data = pd.DataFrame({
                'age': [age], 'job': [job], 'marital': [marital], 'education': [education],
                'default': [default], 'housing': [housing], 'loan': [loan],
                'contact': [contact], 'month': [month], 'day_of_week': [day_of_week],
                'duration': [duration], 'campaign': [campaign], 'pdays': [pdays],
                'previous': [previous], 'poutcome': [poutcome],
                'emp.var.rate': [emp_var_rate], 'cons.price.idx': [cons_price_idx],
                'cons.conf.idx': [cons_conf_idx], 'euribor3m': [euribor3m],
                'nr.employed': [nr_employed]
            })
            
            try:
                prediction = model.predict(input_data)[0]
                prediction_proba = model.predict_proba(input_data)[0]
                
                st.markdown("---")
                st.markdown("### 🎯 Prediction Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.success("### ✅ HIGH LIKELIHOOD")
                        st.markdown("**Customer likely to subscribe**")
                    else:
                        st.error("### ❌ LOW LIKELIHOOD")
                        st.markdown("**Customer unlikely to subscribe**")
                    
                    st.divider()
                    st.metric("Subscription Probability", f"{prediction_proba[1]*100:.1f}%")
                    st.metric("Non-Subscription Probability", f"{prediction_proba[0]*100:.1f}%")
                
                with col2:
                    st.markdown("#### Confidence Meter")
                    st.markdown("**Will Subscribe:**")
                    st.progress(prediction_proba[1])
                    st.markdown(f"<p style='font-size: 24px; font-weight: bold; color: {'green' if prediction_proba[1] > 0.5 else 'orange'};'>{prediction_proba[1]*100:.1f}%</p>", unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    st.markdown("**Will NOT Subscribe:**")
                    st.progress(prediction_proba[0])
                    st.markdown(f"<p style='font-size: 24px; font-weight: bold; color: {'red' if prediction_proba[0] > 0.5 else 'orange'};'>{prediction_proba[0]*100:.1f}%</p>", unsafe_allow_html=True)
                    
                    confidence = max(prediction_proba)
                    st.markdown("---")
                    st.markdown("**Model Confidence:**")
                    if confidence > 0.8:
                        st.success(f"🟢 Very High: {confidence*100:.1f}%")
                    elif confidence > 0.65:
                        st.info(f"🔵 High: {confidence*100:.1f}%")
                    else:
                        st.warning(f"🟡 Moderate: {confidence*100:.1f}%")
                
                st.markdown("### 💡 Key Insights")
                insights = []
                if duration > 300:
                    insights.append("✅ **Long call duration** - Strong positive indicator")
                if poutcome == "success":
                    insights.append("✅ **Previous campaign success**")
                if campaign > 5:
                    insights.append("⚠️ **High contact frequency**")
                
                for insight in insights:
                    st.markdown(insight)
                if not insights:
                    st.info("Standard customer profile")
                    
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    with tab2:
        st.markdown("### 📁 Batch Prediction")
        st.info("Upload CSV file with multiple customer records")
        
        with st.expander("📋 CSV Format"):
            st.markdown("**Columns:** age;job;marital;education;default;housing;loan;contact;month;day_of_week;duration;campaign;pdays;previous;poutcome;emp.var.rate;cons.price.idx;cons.conf.idx;euribor3m;nr.employed")
        
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file, sep=';')
            st.success(f"✅ {len(df)} records loaded")
            
            with st.expander("👀 Preview"):
                st.dataframe(df.head(10))
            
            if st.button("🚀 Run Predictions", type="primary"):
                predictions = model.predict(df)
                probabilities = model.predict_proba(df)
                
                df['prediction'] = predictions
                df['prediction_label'] = df['prediction'].map({0: 'No', 1: 'Yes'})
                df['probability_subscribe'] = probabilities[:, 1]
                df['confidence'] = probabilities.max(axis=1)
                
                st.markdown("### 📊 Summary")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total", len(df))
                with col2:
                    st.metric("YES", predictions.sum())
                with col3:
                    st.metric("NO", len(df) - predictions.sum())
                with col4:
                    st.metric("Avg Confidence", f"{df['confidence'].mean()*100:.1f}%")
                
                st.dataframe(df)
                
                csv = df.to_csv(index=False, sep=';')
                st.download_button("📥 Download Results", csv, f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    with tab3:
        st.markdown("### 📈 Model Performance")
        import glob, json
        summary_files = glob.glob("outputs/summary_*.json")
        
        if summary_files:
            with open(summary_files[-1]) as f:
                summary = json.load(f)
            
            data = []
            for name, metrics in summary.items():
                if 'accuracy' in metrics:
                    data.append({
                        'Model': name,
                        'Accuracy': f"{metrics['accuracy']*100:.2f}%",
                        'Precision': f"{metrics['precision']*100:.2f}%",
                        'Recall': f"{metrics['recall']*100:.2f}%",
                        'F1-Score': f"{metrics['f1']*100:.2f}%"
                    })
            
            st.dataframe(pd.DataFrame(data), use_container_width=True)
        else:
            st.info("Train models first using `tes_full.py`")

else:
    st.warning("⚠️ No model loaded")
    st.markdown("Train models: `python tes_full.py --data bank-additional-full.csv`")

# Footer
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Built with:** Streamlit")
with col2:
    st.markdown("**ML Framework:** scikit-learn")
with col3:
    st.markdown("**Dataset:** UCI Bank Marketing")

st.caption("© 2024 Bank Marketing Predictor | Powered by Machine Learning")
