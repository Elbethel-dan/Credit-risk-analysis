import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM CSS (Professional Styling)
# -------------------------------------------------
st.markdown("""
<style>
.big-title {
    font-size:32px !important;
    font-weight:700;
}
.section-title {
    font-size:20px !important;
    font-weight:600;
    margin-top:20px;
}
.card {
    padding:20px;
    border-radius:12px;
    background-color:#f8f9fa;
    box-shadow:0px 4px 12px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("/Users/elbethelzewdie/Downloads/credit-risk-analysis/Credit-risk-analysis/model/LogisticRegression_best_model1.pkl")

model = load_model()

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown('<p class="big-title">🛡️ Credit Risk Prediction Dashboard</p>', unsafe_allow_html=True)
st.write("This system predicts whether a customers credit risk.")

st.divider()

# -------------------------------------------------
# FEATURE MAPPING
# -------------------------------------------------

feature_mapping = {
    "CustomerId": "Customer ID",
    "Amount_sum": "Total Transaction Amount",
    "Amount_avg": "Average Transaction Amount",
    "Amount_count": "Number of Transactions",
    "Amount_std": "Transaction Amount Variability",
    "Value_sum": "Total Value",
    "Value_avg": "Average Value",
    "Value_count": "Number of Value Entries",
    "Value_std": "Value Variability",
    "ProviderId_mode": "Most Used Provider ID",
    "ProductId_mode": "Most Used Product ID",
    "ProductCategory_mode": "Most Used Product Category",
    "ChannelId_mode": "Most Used Channel",
    "transaction_hour_avg": "Average Transaction Hour",
    "transaction_day_avg": "Average Transaction Day",
    "transaction_month_avg": "Average Transaction Month",
    "transaction_year_avg": "Average Transaction Year",
    "transaction_dayofweek_avg": "Average Day of Week",
    "is_weekend_avg": "Weekend Transaction Ratio",
    "is_business_hours_avg": "Business Hours Transaction Ratio"
}

expected_features = list(feature_mapping.keys())

# -------------------------------------------------
# SIDEBAR INPUT
# -------------------------------------------------
st.sidebar.header("Enter Customer Transaction Details")

inputs = {}

for feature, label in feature_mapping.items():
    if "avg" in feature or "std" in feature:
        inputs[feature] = st.sidebar.number_input(label, value=0.0)
    else:
        inputs[feature] = st.sidebar.number_input(label, value=0)

input_df = pd.DataFrame([inputs], columns=expected_features)

# -------------------------------------------------
# MAIN LAYOUT
# -------------------------------------------------
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">📊 Input Summary</p>', unsafe_allow_html=True)
    st.dataframe(input_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">🔎 Fraud Prediction</p>', unsafe_allow_html=True)

    if st.button("Run Fraud Analysis"):

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={'text': "Fraud Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red" if probability > 0.5 else "green"},
                'steps': [
                    {'range': [0, 50], 'color': "#d4edda"},
                    {'range': [50, 100], 'color': "#f8d7da"}
                ],
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

        if prediction == 1:
            st.error("⚠️ High Credit Risk Detected")
        else:
            st.success("✅ Low Credit Risk Detected")

    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

st.caption("© Credit Risk Analysis | ML Model Powered by Scikit-Learn")
