import streamlit as st
import numpy as np
import joblib

# -----------------------------
# Load Model and Scaler
# -----------------------------
MODEL_PATH = "heart_disease_rf_model.pkl"
SCALER_PATH = "scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# -----------------------------
# Title
# -----------------------------
st.title("❤️ Heart Disease Risk Assessment AI Agent")

st.write("Enter patient details below")

# -----------------------------
# Inputs
# -----------------------------

age = st.number_input("Age", 1, 100, 50)

sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

cp = st.selectbox(
    "Chest Pain Type (cp)",
    [0, 1, 2, 3]
)

trestbps = st.number_input(
    "Resting Blood Pressure",
    80,
    250,
    120
)

chol = st.number_input(
    "Cholesterol",
    100,
    600,
    200
)

fbs = st.selectbox(
    "Fasting Blood Sugar > 120",
    ["Yes", "No"]
)

restecg = st.selectbox(
    "Rest ECG",
    [0, 1, 2]
)

thalach = st.number_input(
    "Maximum Heart Rate",
    50,
    250,
    150
)

exang = st.selectbox(
    "Exercise Induced Angina",
    ["Yes", "No"]
)

oldpeak = st.number_input(
    "Oldpeak",
    0.0,
    10.0,
    1.0
)

slope = st.selectbox(
    "Slope",
    [0, 1, 2]
)

ca = st.selectbox(
    "Major Vessels (ca)",
    [0, 1, 2, 3]
)

thal = st.selectbox(
    "Thal",
    [0, 1, 2, 3]
)

# -----------------------------
# Encoding
# -----------------------------

sex = 1 if sex == "Male" else 0

fbs = 1 if fbs == "Yes" else 0

exang = 1 if exang == "Yes" else 0

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Heart Disease Risk"):

    input_data = np.array([
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]).reshape(1, -1)

    # Debug
    st.write("Input Data:")
    st.write(input_data)

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    probability = model.predict_proba(input_scaled)

    risk = probability[0][1] * 100

    st.subheader("Prediction Result")

    st.write("Prediction Value:", int(prediction[0]))
    st.write("Probability:", probability)

    if prediction[0] == 1:

        st.error("⚠️ Heart Disease Detected")

    else:

        st.success("✅ No Heart Disease Detected")

    st.write(
        f"Heart Disease Risk Probability: {risk:.2f}%"
    )

    # -----------------------------
    # Recommendation
    # -----------------------------

    st.subheader("Medical Recommendation")

    if risk >= 70:

        st.warning(
            "Please consult a cardiologist immediately."
        )

    elif risk >= 40:

        st.info(
            "Moderate Risk. Regular checkups and healthy lifestyle are recommended."
        )

    else:

        st.success(
            "Low Risk. Continue healthy habits."
        )

    # -----------------------------
    # AI Health Assistant
    # -----------------------------

    st.subheader("🤖 AI Health Assistant")

    if prediction[0] == 1:

        st.write(f"""
        Estimated Heart Disease Risk: {risk:.2f}%

        Possible Risk Factors:
        • Cholesterol = {chol}
        • Blood Pressure = {trestbps}
        • Exercise Angina = {exang}

        Recommendations:
        • Consult a Cardiologist
        • Reduce Salt Intake
        • Follow a Heart Healthy Diet
        • Exercise Regularly
        • Monitor BP and Cholesterol
        """)

    else:

        st.write(f"""
        Estimated Heart Disease Risk: {risk:.2f}%

        Recommendations:
        • Continue Healthy Lifestyle
        • Maintain Healthy Weight
        • Regular Exercise
        • Routine Medical Checkups
        """)

# -----------------------------
# AI Chat Agent
# -----------------------------

st.subheader("💬 Ask the AI Health Agent")

question = st.text_input(
    "Ask a Heart Health Question"
)

if question:

    q = question.lower()

    if "cholesterol" in q:

        st.write(
            "High cholesterol can increase the risk of heart disease. A healthy diet and exercise can help lower it."
        )

    elif "blood pressure" in q:

        st.write(
            "Maintaining normal blood pressure reduces strain on the heart."
        )

    elif "exercise" in q:

        st.write(
            "Regular exercise improves cardiovascular health and reduces risk."
        )

    elif "heart disease" in q:

        st.write(
            "Heart disease refers to conditions affecting the heart and blood vessels."
        )

    else:

        st.write(
            "Please consult a healthcare professional for personalized medical advice."
        )
