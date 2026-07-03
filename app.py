import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Health Condition Predictor",
    page_icon="🩺",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("health_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Title
# -----------------------------
st.title("🩺 Health Condition Prediction")
st.markdown(
    "Enter your health information below to predict your health condition."
)

st.divider()

# -----------------------------
# User Inputs
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    sleep_duration = st.number_input(
        "Sleep Duration (Hours)",
        min_value=0.0,
        max_value=24.0,
        value=7.0
    )

    heart_rate = st.number_input(
        "Heart Rate",
        value=75.0
    )

    bmi = st.number_input(
        "BMI",
        value=23.0
    )

    calorie_expenditure = st.number_input(
        "Calorie Expenditure",
        value=2200.0
    )

with col2:
    step_count = st.number_input(
        "Step Count",
        value=8000
    )

    exercise_duration = st.number_input(
        "Exercise Duration (Minutes)",
        value=40.0
    )

    water_intake = st.number_input(
        "Water Intake (Litres)",
        value=2.0
    )

st.divider()

diet = st.selectbox(
    "Diet Type",
    ["balanced", "non-veg", "veg"]
)

stress = st.selectbox(
    "Stress Level",
    ["low", "medium", "high"]
)

sleep_quality = st.selectbox(
    "Sleep Quality",
    ["poor", "average", "good", "excellent"]
)

activity = st.selectbox(
    "Physical Activity",
    ["sedentary", "moderate", "active"]
)

smoking = st.selectbox(
    "Smoking / Alcohol",
    ["no", "yes", "occasional"]
)

gender = st.selectbox(
    "Gender",
    ["female", "male", "other"]
)

# -----------------------------
# Manual Encoding
# -----------------------------

stress_map = {
    "low": 0,
    "medium": 1,
    "high": 2
}

sleep_map = {
    "poor": 0,
    "average": 1,
    "good": 2,
    "excellent": 3
}

activity_map = {
    "sedentary": 0,
    "moderate": 1,
    "active": 2
}

data = {
    "sleep_duration": sleep_duration,
    "heart_rate": heart_rate,
    "bmi": bmi,
    "calorie_expenditure": calorie_expenditure,
    "step_count": step_count,
    "exercise_duration": exercise_duration,
    "water_intake": water_intake,

    "stress_level": stress_map[stress],
    "sleep_quality": sleep_map[sleep_quality],
    "physical_activity_level": activity_map[activity],

    "diet_type_balanced": 1 if diet == "balanced" else 0,
    "diet_type_non-veg": 1 if diet == "non-veg" else 0,
    "diet_type_veg": 1 if diet == "veg" else 0,

    "gender_female": 1 if gender == "female" else 0,
    "gender_male": 1 if gender == "male" else 0,
    "gender_other": 1 if gender == "other" else 0,

    "smoking_alcohol_no": 1 if smoking == "no" else 0,
    "smoking_alcohol_occasional": 1 if smoking == "occasional" else 0,
    "smoking_alcohol_yes": 1 if smoking == "yes" else 0,
}

input_df = pd.DataFrame([data])

# -----------------------------
# Scale Features
# -----------------------------
scaled_input = scaler.transform(input_df)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Health Condition", use_container_width=True):

    prediction = model.predict(scaled_input)[0]

    labels = {
        0: "At-Risk",
        1: "Fit",
        2: "Unhealthy"
    }

    result = labels[prediction]

    st.divider()

    if result == "Fit":
        st.success(f"✅ Prediction: **{result}**")

    elif result == "At-Risk":
        st.warning(f"⚠️ Prediction: **{result}**")

    else:
        st.error(f"🚨 Prediction: **{result}**")