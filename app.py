import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Diabetes Risk Screening Tool",
    layout="centered"
)

# --------------------------------------------------
# Load artifacts
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("diabetes_model.joblib")
    model_bal = joblib.load("diabetes_model_balanced.joblib")
    scaler = joblib.load("scaler.joblib")
    features = joblib.load("features.joblib")
    return model, model_bal, scaler, features

model, model_bal, scaler, features = load_artifacts()

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown("## 🩺 Diabetes Risk Screening Tool")

st.markdown(
    "This tool estimates the risk of diabetes based on common health and lifestyle factors."
)

st.warning(
    "⚠️ The result is for screening purposes only and does not provide a medical diagnosis."
)

# --------------------------------------------------
# Patient information
# --------------------------------------------------
st.markdown("### Patient information")

age = st.number_input(
    "Age (years)",
    min_value=18,
    max_value=100,
    value=61,
    step=1
)

sex = st.selectbox(
    "Sex",
    ["Female", "Male"]
)

education = st.selectbox(
    "Education level",
    ["Low", "Medium", "High"]
)

marital = st.selectbox(
    "Marital status",
    ["Single", "Married"]
)

labor = st.selectbox(
    "Labor status",
    ["Unemployed", "Employed"]
)

smoking = st.selectbox(
    "Smoking",
    ["No", "Yes"]
)

alcohol = st.selectbox(
    "Alcohol drinking",
    ["No", "Yes"]
)

physical = st.selectbox(
    "Physical inactivity",
    ["No", "Yes"]
)

salt = st.selectbox(
    "High salt intake",
    ["No", "Yes"]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=30.0,
    step=0.1
)

waist = st.number_input(
    "Waist circumference (cm)",
    min_value=50,
    max_value=150,
    value=90,
    step=1
)

# --------------------------------------------------
# Screening mode (hidden simple logic)
# --------------------------------------------------
# By default: high sensitivity for screening
clf = model_bal
threshold = 0.3

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("Predict risk"):

    input_df = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "Education_level": education,
        "Marital_status": marital,
        "Labor_status": labor,
        "Smoking": smoking,
        "Alcohol_drinking": alcohol,
        "Physical_inactivity": physical,
        "High_salt_intake": salt,
        "BMI": bmi,
        "Waist_circumference": waist
    }])

    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=features, fill_value=0)
    input_scaled = scaler.transform(input_df)

    risk = clf.predict_proba(input_scaled)[0][1]

    st.markdown("### Result")
    st.write(f"Estimated probability of diabetes: **{risk:.2f}**")

    if risk >= threshold:
        st.error("⚠️ High risk of diabetes (screen-positive)")
    else:
        st.success("✅ Low risk of diabetes (screen-negative)")
# (вставьте сюда код Streamlit-приложения)
