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

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 5.5rem;
        font-weight: 900;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        margin-top: 0;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.15);
        line-height: 1.1;
        letter-spacing: -1px;
    }
    .sub-header {
        font-size: 1.4rem;
        color: #555;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 400;
        line-height: 1.5;
    }
    h3 {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        color: #2c3e50;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
    h4 {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: #34495e;
        margin-top: 1rem !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .info-box {
        background-color: #f0f7ff;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: 600;
        padding: 0.75rem;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #1557b0;
        border-color: #1557b0;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model(model_path):
    """Load the trained model pipeline"""
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Helper dictionaries for user-friendly labels
JOB_LABELS = {
    "admin.": "Administrative",
    "blue-collar": "Blue-collar Worker",
    "entrepreneur": "Entrepreneur",
    "housemaid": "Housemaid",
    "management": "Management",
    "retired": "Retired",
    "self-employed": "Self-employed",
    "services": "Services",
    "student": "Student",
    "technician": "Technician",
    "unemployed": "Unemployed",
    "unknown": "Unknown"
}

EDUCATION_LABELS = {
    "basic.4y": "Basic Education (4 years)",
    "basic.6y": "Basic Education (6 years)",
    "basic.9y": "Basic Education (9 years)",
    "high.school": "High School",
    "illiterate": "Illiterate",
    "professional.course": "Professional Course",
    "university.degree": "University Degree",
    "unknown": "Unknown"
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

# Header
st.markdown('<p class="main-header">🏦 Bank Marketing Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict customer subscription likelihood for term deposits using machine learning</p>', unsafe_allow_html=True)

# Sidebar - Model Selection
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-building.png", width=80)
    st.header("⚙️ Model Configuration")
    
    model_options = {
        "🌲 Random Forest": "outputs/pipeline_RandomForest.joblib",
        "🌲 Random Forest (Optimized)": "outputs/RandomForest_GridSearch.joblib",
        "📊 Logistic Regression": "outputs/pipeline_LogisticRegression.joblib",
        "🌳 Decision Tree": "outputs/pipeline_DecisionTree.joblib",
    }
    
    selected_model_display = st.selectbox(
        "Select ML Model:",
        list(model_options.keys()),
        help="Choose the machine learning model for prediction"
    )
    
    model_path = model_options[selected_model_display]
    model = load_model(model_path)
    
    if model is not None:
        st.success(f"✅ Model loaded successfully!")
        
        # Model info
        with st.expander("📋 Model Information"):
            st.markdown(f"""
            **Model:** {selected_model_display}
            
            **Training Details:**
            - Preprocessing: OneHotEncoding + StandardScaler
            - Class Balancing: SMOTE
            - Trained on: UCI Bank Marketing Dataset
            """)
    else:
        st.error("❌ Model not found")
        st.info("Please train models first using `tes_full.py`")
    
    st.divider()
    
    # Quick Stats
    st.subheader("📊 Quick Stats")
    st.metric("Models Available", len([p for p in model_options.values() if Path(p).exists()]))
    st.metric("Last Updated", datetime.now().strftime("%Y-%m-%d"))

# Main content
if model is not None:
    # Tabs for better organization
    tab1, tab2, tab3 = st.tabs(["🎯 Single Prediction", "📁 Batch Prediction", "📈 Model Performance"])
    
    # TAB 1: Single Prediction
    with tab1:
        st.markdown("### Customer Information")
        
        # Customer Demographics
        with st.container():
            st.markdown("#### 👤 Demographics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.number_input(
                    "Age",
                    min_value=18,
                    max_value=100,
                    value=35,
                    help="Customer's age in years"
                )
            
            with col2:
                job_display = st.selectbox(
                    "Occupation",
                    list(JOB_LABELS.values()),
                    help="Customer's job type"
                )
                job = [k for k, v in JOB_LABELS.items() if v == job_display][0]
            
            with col3:
                marital_display = st.selectbox(
                    "Marital Status",
                    ["Married", "Single", "Divorced", "Unknown"]
                )
                marital = marital_display.lower()
        
        col1, col2 = st.columns(2)
        
        with col1:
            education_display = st.selectbox(
                "Education Level",
                list(EDUCATION_LABELS.values()),
                help="Highest education level achieved"
            )
            education = [k for k, v in EDUCATION_LABELS.items() if v == education_display][0]
        
        with col2:
            st.write("")  # Spacing
        
        st.divider()
        
        # Financial Information
        st.markdown("#### 💳 Financial Profile")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            default_display = st.selectbox(
                "Credit in Default?",
                ["No", "Yes", "Unknown"],
                help="Has credit in default?"
            )
            default = default_display.lower()
        
        with col2:
            housing_display = st.selectbox(
                "Housing Loan?",
                ["No", "Yes", "Unknown"],
                help="Has housing loan?"
            )
            housing = housing_display.lower()
        
        with col3:
            loan_display = st.selectbox(
                "Personal Loan?",
                ["No", "Yes", "Unknown"],
                help="Has personal loan?"
            )
            loan = loan_display.lower()
        
        st.divider()
        
        # Campaign Information
        st.markdown("#### 📞 Campaign Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            contact_display = st.selectbox(
                "Contact Method",
                ["Cellular", "Telephone"],
                help="Communication type used"
            )
            contact = contact_display.lower()
        
        with col2:
            month_display = st.selectbox(
                "Last Contact Month",
                list(MONTH_LABELS.values()),
                help="Month of last contact"
            )
            month = [k for k, v in MONTH_LABELS.items() if v == month_display][0]
        
        with col3:
            day_display = st.selectbox(
                "Last Contact Day",
                list(DAY_LABELS.values()),
                help="Day of week of last contact"
            )
            day_of_week = [k for k, v in DAY_LABELS.items() if v == day_display][0]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            duration = st.number_input(
                "Call Duration (seconds)",
                min_value=0,
                max_value=5000,
                value=180,
                help="Duration of last contact in seconds"
            )
            st.caption(f"≈ {duration // 60} minutes {duration % 60} seconds")
        
        with col2:
            campaign = st.number_input(
                "Contacts This Campaign",
                min_value=1,
                max_value=50,
                value=2,
                help="Number of contacts during this campaign"
            )
        
        with col3:
            previous = st.number_input(
                "Previous Contacts",
                min_value=0,
                max_value=10,
                value=0,
                help="Number of contacts before this campaign"
            )
        
        col1, col2 = st.columns(2)
        
        with col1:
            pdays = st.number_input(
                "Days Since Last Contact",
                min_value=0,
                max_value=999,
                value=999,
                help="Days since last contact from previous campaign (999 = not contacted)"
            )
            if pdays == 999:
                st.caption("⚠️ Not previously contacted")
            else:
                st.caption(f"≈ {pdays // 30} months ago")
        
        with col2:
            poutcome_display = st.selectbox(
                "Previous Campaign Outcome",
                ["Non-existent", "Failure", "Success"],
                help="Outcome of previous marketing campaign"
            )
            poutcome = poutcome_display.lower().replace("-", "")
        
        # Economic Indicators (Collapsible)
        with st.expander("📊 Economic Indicators (Advanced Settings)"):
            st.info("These indicators reflect macroeconomic conditions. Default values are based on dataset averages.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                emp_var_rate = st.number_input(
                    "Employment Variation Rate",
                    value=1.1,
                    format="%.2f",
                    help="Employment variation rate - quarterly indicator"
                )
                
                cons_price_idx = st.number_input(
                    "Consumer Price Index",
                    value=93.994,
                    format="%.3f",
                    help="Consumer price index - monthly indicator"
                )
                
                cons_conf_idx = st.number_input(
                    "Consumer Confidence Index",
                    value=-36.4,
                    format="%.1f",
                    help="Consumer confidence index - monthly indicator"
                )
            
            with col2:
                euribor3m = st.number_input(
                    "Euribor 3 Month Rate",
                    value=4.857,
                    format="%.3f",
                    help="Euribor 3 month rate - daily indicator"
                )
                
                nr_employed = st.number_input(
                    "Number of Employees (thousands)",
                    value=5191.0,
                    format="%.1f",
                    help="Number of employees - quarterly indicator"
                )
        
        st.divider()
        
        # Predict button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_button = st.button("🔮 Predict Subscription Likelihood", type="primary", use_container_width=True)
        
        if predict_button:
            # Create input dataframe
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
            
            try:
                with st.spinner("Analyzing customer profile..."):
                    prediction = model.predict(input_data)[0]
                    prediction_proba = model.predict_proba(input_data)[0]
                
                st.markdown("---")
                st.markdown("### 🎯 Prediction Results")
                
                # Results with gauge chart
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    if prediction == 1:
                        st.success("### ✅ HIGH LIKELIHOOD")
                        st.markdown("**Customer is likely to subscribe to term deposit**")
                    else:
                        st.error("### ❌ LOW LIKELIHOOD")
                        st.markdown("**Customer is unlikely to subscribe to term deposit**")
                    
                    st.divider()
                    
                    # Probability metrics
                    st.metric(
                        "Subscription Probability",
                        f"{prediction_proba[1]*100:.1f}%",
                        delta=f"{(prediction_proba[1] - 0.5)*100:.1f}% vs. baseline"
                    )
                    
                    st.metric(
                        "Non-Subscription Probability",
                        f"{prediction_proba[0]*100:.1f}%"
                    )
                
                with col2:
                    # Simple progress bar visualization
                    st.markdown("#### Confidence Meter")
                    
                    # Create visual representation with colored boxes
                    prob_yes = prediction_proba[1] * 100
                    prob_no = prediction_proba[0] * 100
                    
                    # Yes probability bar
                    st.markdown("**Will Subscribe:**")
                    st.progress(prediction_proba[1])
                    st.markdown(f"<p style='font-size: 24px; font-weight: bold; color: {'green' if prob_yes > 50 else 'orange'};'>{prob_yes:.1f}%</p>", unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # No probability bar
                    st.markdown("**Will NOT Subscribe:**")
                    st.progress(prediction_proba[0])
                    st.markdown(f"<p style='font-size: 24px; font-weight: bold; color: {'red' if prob_no > 50 else 'orange'};'>{prob_no:.1f}%</p>", unsafe_allow_html=True)
                    
                    # Confidence level indicator
                    confidence = max(prediction_proba)
                    st.markdown("---")
                    st.markdown("**Model Confidence:**")
                    
                    if confidence > 0.8:
                        st.success(f"🟢 Very High: {confidence*100:.1f}%")
                    elif confidence > 0.65:
                        st.info(f"🔵 High: {confidence*100:.1f}%")
                    elif confidence > 0.55:
                        st.warning(f"🟡 Moderate: {confidence*100:.1f}%")
                    else:
                        st.error(f"🔴 Low: {confidence*100:.1f}%")
                
                # Insights
                st.markdown("### 💡 Key Insights")
                
                insights = []
                
                if duration > 300:
                    insights.append("✅ **Long call duration** - Strong positive indicator")
                elif duration < 120:
                    insights.append("⚠️ **Short call duration** - May indicate low interest")
                
                if poutcome == "success":
                    insights.append("✅ **Previous campaign success** - Customer has history of positive response")
                elif poutcome == "failure":
                    insights.append("⚠️ **Previous campaign failure** - Customer previously declined")
                
                if previous > 0 and pdays < 999:
                    insights.append("✅ **Previously contacted** - Customer is familiar with campaigns")
                
                if campaign > 5:
                    insights.append("⚠️ **High contact frequency** - May lead to campaign fatigue")
                
                if age > 60 and job == "retired":
                    insights.append("💰 **Retired demographic** - Potentially stable income for deposits")
                
                for insight in insights:
                    st.markdown(insight)
                
                if not insights:
                    st.info("Standard customer profile - prediction based on comprehensive feature analysis.")
                
            except Exception as e:
                st.error(f"❌ Prediction error: {e}")
                st.info("Please ensure all input values are valid.")
    
    # TAB 2: Batch Prediction
    with tab2:
        st.markdown("### 📁 Batch Prediction")
        st.info("Upload a CSV file with multiple customer records for bulk predictions. The file should have the same format as the training data.")
        
        # File format guide
        with st.expander("📋 CSV Format Requirements"):
            st.markdown("""
            **Required columns (must include all):**
            - `age`, `job`, `marital`, `education`, `default`, `housing`, `loan`
            - `contact`, `month`, `day_of_week`, `duration`, `campaign`, `pdays`, `previous`, `poutcome`
            - `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`
            
            **Important:**
            - ✅ Use **semicolon (;)** as separator
            - ✅ Column names must match exactly (case-sensitive)
            - ✅ Use raw values (e.g., "admin." not "Administrative")
            - ✅ No missing values allowed
            
            **Example header:**
            ```
            age;job;marital;education;default;housing;loan;contact;month;day_of_week;duration;campaign;pdays;previous;poutcome;emp.var.rate;cons.price.idx;cons.conf.idx;euribor3m;nr.employed
            ```
            
            **Example data:**
            ```
            35;admin.;married;university.degree;no;yes;no;cellular;may;mon;180;2;999;0;nonexistent;1.1;93.994;-36.4;4.857;5191.0
            ```
            """)
        
        st.markdown("#### ⬆️ Upload Your File")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload CSV file with customer data"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, sep=';')
                
                st.success(f"✅ File uploaded successfully! Found {len(df)} records.")
                
                with st.expander("👀 Preview Data (first 10 rows)"):
                    st.dataframe(df.head(10), use_container_width=True)
                
                if st.button("🚀 Run Batch Predictions", type="primary"):
                    with st.spinner("Processing predictions..."):
                        predictions = model.predict(df)
                        probabilities = model.predict_proba(df)
                        
                        df['prediction'] = predictions
                        df['prediction_label'] = df['prediction'].map({0: 'No', 1: 'Yes'})
                        df['probability_subscribe'] = probabilities[:, 1]
                        df['probability_not_subscribe'] = probabilities[:, 0]
                        df['confidence'] = probabilities.max(axis=1)
                    
                    # Summary statistics
                    st.markdown("### 📊 Prediction Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Records", len(df))
                    
                    with col2:
                        st.metric("Predicted: YES", f"{predictions.sum()}", 
                                 delta=f"{predictions.sum()/len(df)*100:.1f}%")
                    
                    with col3:
                        st.metric("Predicted: NO", f"{len(df) - predictions.sum()}",
                                 delta=f"{(len(df) - predictions.sum())/len(df)*100:.1f}%")
                    
                    with col4:
                        st.metric("Avg. Confidence", f"{df['confidence'].mean()*100:.1f}%")
                    
                    # Results table
                    st.markdown("### 📋 Detailed Results")
                    st.dataframe(
                        df.style.format({
                            'probability_subscribe': '{:.2%}',
                            'probability_not_subscribe': '{:.2%}',
                            'confidence': '{:.2%}'
                        }),
                        use_container_width=True
                    )
                    
                    # Download buttons
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        csv = df.to_csv(index=False, sep=';')
                        st.download_button(
                            label="📥 Download Full Results (CSV)",
                            data=csv,
                            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        # High probability customers only
                        high_prob = df[df['probability_subscribe'] > 0.7]
                        csv_high = high_prob.to_csv(index=False, sep=';')
                        st.download_button(
                            label="📥 Download High-Probability Leads (>70%)",
                            data=csv_high,
                            file_name=f"high_probability_leads_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
                st.info("Please ensure your CSV file matches the required format.")
    
    # TAB 3: Model Performance
    with tab3:
        st.markdown("### 📈 Model Performance Metrics")
        
        # Try to load summary JSON
        import glob
        summary_files = glob.glob("outputs/summary_*.json")
        
        if summary_files:
            import json
            with open(summary_files[-1], 'r') as f:
                summary = json.load(f)
            
            # Model comparison
            st.markdown("#### Model Comparison")
            
            model_names = []
            accuracies = []
            precisions = []
            recalls = []
            f1_scores = []
            
            for model_name, metrics in summary.items():
                if 'accuracy' in metrics:
                    model_names.append(model_name)
                    accuracies.append(metrics['accuracy'] * 100)
                    precisions.append(metrics['precision'] * 100)
                    recalls.append(metrics['recall'] * 100)
                    f1_scores.append(metrics['f1'] * 100)
            
            # Create comparison dataframe
            comparison_df = pd.DataFrame({
                'Model': model_names,
                'Accuracy': accuracies,
                'Precision': precisions,
                'Recall': recalls,
                'F1-Score': f1_scores
            })
            
            st.dataframe(
                comparison_df.style.format({
                    'Accuracy': '{:.2f}%',
                    'Precision': '{:.2f}%',
                    'Recall': '{:.2f}%',
                    'F1-Score': '{:.2f}%'
                }).highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score']),
                use_container_width=True
            )
        
        else:
            st.info("No performance metrics found. Train models using `tes_full.py` to generate performance reports.")

else:
    # Model not loaded
    st.warning("⚠️ No model loaded")
    
    st.markdown("""
    ### 🚀 Getting Started
    
    To use this application, you need to train the machine learning models first:
    
    **Step 1: Train Models**
    ```bash
    python tes_full.py --data bank-additional-full.csv
    ```
    
    **Step 2: Verify Output**
    - Models will be saved in `outputs/` folder
    - Look for `.joblib` files
    
    **Step 3: Refresh**
    - Refresh this page to load the trained models
    
    **Step 4: Start Predicting!**
    - Use the Single Prediction tab for individual customers
    - Use Batch Prediction for multiple customers at once
    """)

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
