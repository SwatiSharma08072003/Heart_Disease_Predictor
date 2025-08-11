import streamlit as st
from model_training import x_train_acc, x_test_acc

with st.sidebar:
    st.markdown("## 🩺 Heart Disease Predictor")
    st.image("assets/heart.png", width=200)
    st.markdown("### 📘 About This App")
    st.write(
        "This tool uses a machine learning model to predict the likelihood of heart disease "
        "based on patient health metrics. Please enter the required values in the main panel."
    )

    st.markdown("### 📋 Expected Features")
    st.markdown("""
    - **Age**: Age in years  
    - **Sex**: 0 = Female, 1 = Male  
    - **CP**: Chest pain type (0–3)  
    - **Trestbps**: Resting blood pressure  
    - **Chol**: Serum cholesterol  
    - **FBS**: Fasting blood sugar > 120 mg/dl (0/1)  
    - **Restecg**: Resting ECG results (0–2)  
    - **Thalach**: Max heart rate achieved  
    - **Exang**: Exercise-induced angina (0/1)  
    - **Oldpeak**: ST depression  
    - **Slope**: Slope of peak exercise ST segment (0–2)  
    - **CA**: Number of major vessels (0–3)  
    - **Thal**: Thalassemia (1–3)
    """)

    st.markdown("### 📊 Model Performance")
    st.metric("Training Accuracy", f"{x_train_acc:.2f}")
    st.metric("Test Accuracy", f"{x_test_acc:.2f}")
    st.caption("Model: Logistic Regression with Standard Scaling")

    st.markdown("---")
    st.markdown("💡 Tip: Use realistic values for better predictions.")
    st.markdown("🧠 Powered by Scikit-learn")