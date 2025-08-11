import streamlit as st
import pandas as pd
from data_preprocessing import x, scaler
from model_training import model_scaled

st.title("💓 Heart Disease Prediction System")
st.caption(f"Note: Enter values for the following {x.shape[1]} features.")
st.subheader("🧾 Enter Patient Details:")

user_input = {}
user_input["age"] = st.number_input("Age", min_value=1, max_value=120, value=50)
user_input["sex"] = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
user_input["cp"] = st.selectbox("Chest Pain Type (CP)", options=[0, 1, 2, 3])
user_input["trestbps"] = st.number_input("Resting Blood Pressure (Trestbps)", min_value=80, max_value=200, value=120)
user_input["chol"] = st.number_input("Serum Cholesterol (Chol)", min_value=100, max_value=600, value=240)
user_input["fbs"] = st.selectbox("Fasting Blood Sugar > 120 mg/dl (FBS)", options=[0, 1])
user_input["restecg"] = st.selectbox("Resting ECG Results (Restecg)", options=[0, 1, 2])
user_input["thalach"] = st.number_input("Max Heart Rate Achieved (Thalach)", min_value=60, max_value=220, value=150)
user_input["exang"] = st.selectbox("Exercise-Induced Angina (Exang)", options=[0, 1])
user_input["oldpeak"] = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
user_input["slope"] = st.selectbox("Slope of Peak Exercise ST Segment (Slope)", options=[0, 1, 2])
user_input["ca"] = st.selectbox("Number of Major Vessels (CA)", options=[0, 1, 2, 3])
user_input["thal"] = st.selectbox("Thalassemia (Thal)", options=[1, 2, 3])

input_df = pd.DataFrame([user_input])
input_df.columns = x.columns
input_scaled_df = pd.DataFrame(scaler.transform(input_df), columns=x.columns)

if st.button("Predict"):
    prediction = model_scaled.predict(input_scaled_df)
    probability = model_scaled.predict_proba(input_scaled_df)[0][1]

    st.markdown("---")
    st.subheader("🧠 Prediction Result")
    if prediction[0] == 1:
        st.error("⚠️ You may be at risk of heart disease. Please consult a healthcare professional.")
    else:
        st.success("✅ Your report appears normal. Keep up the healthy lifestyle!")

    st.write(f"📊 Probability of Heart Disease: **{probability:.2f}**")
    st.progress(int(probability * 100))

    if probability > 0.75:
        st.warning("🔎 High confidence in prediction.")
    elif probability > 0.5:
        st.info("🔎 Moderate confidence in prediction.")
    else:
        st.info("🔎 Low confidence in prediction.")

st.markdown("---")
st.markdown("## 🙏 Thanks for visiting!")
st.markdown("### Stay healthy and take care! 💖")