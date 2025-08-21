import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

ARTIFACTS_DIR = "artifacts"

# --- Safe Loader Functions ---
def safe_load_joblib(path):
    if not os.path.exists(path):
        st.error(f"❌ Missing required file: {path}. Please upload the artifacts folder with trained models.")
        st.stop()
    return joblib.load(path)

def safe_load_csv(path):
    if not os.path.exists(path):
        st.error(f"❌ Missing required file: {path}. Please upload the artifacts folder with metrics.csv.")
        st.stop()
    return pd.read_csv(path, index_col=0)

# --- Load Required Artifacts ---
scaler = safe_load_joblib(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
metrics_df = safe_load_csv(os.path.join(ARTIFACTS_DIR, "metrics.csv"))

model_files = {
    name.replace("_", " ").title(): os.path.join(ARTIFACTS_DIR, file)
    for name, file in [
        (f.replace(".pkl", ""), f)
        for f in os.listdir(ARTIFACTS_DIR)
        if f.endswith(".pkl") and f != "scaler.pkl"
    ]
}

# --- Streamlit App Config ---
st.set_page_config(page_title="🩺 Diabetic Retinopathy Predictor", layout="centered")
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🩺 Diabetic Retinopathy Prediction</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Model Settings")
if not model_files:
    st.error("❌ No trained model files found in artifacts/. Please upload model .pkl files.")
    st.stop()

model_choice = st.sidebar.selectbox("Choose a Model", list(model_files.keys()))
model = joblib.load(model_files[model_choice])

# --- Input Form ---
st.subheader("Enter Patient Details")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=40)
    systolic_bp = st.number_input("Systolic BP", min_value=50, max_value=250, value=120)
with col2:
    diastolic_bp = st.number_input("Diastolic BP", min_value=30, max_value=150, value=80)
    cholesterol = st.number_input("Cholesterol", min_value=50, max_value=400, value=180)

# --- Prediction Button ---
if st.button("🔮 Predict"):
    input_data = np.array([[age, systolic_bp, diastolic_bp, cholesterol]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    # Risk-based recommendations
    if prediction == 1:
        if prob >= 0.7:
            st.error(f"🚨 High Risk of Retinopathy ({prob*100:.1f}%). Immediate specialist consultation advised.")
        elif prob >= 0.3:
            st.warning(f"⚠️ Moderate Risk of Retinopathy ({prob*100:.1f}%). Follow up with doctor within 6 months.")
        else:
            st.info(f"ℹ️ Possible Risk of Retinopathy ({prob*100:.1f}%). Routine check-up recommended.")
    else:
        st.success(f"✅ Low Risk of Retinopathy ({(1-prob)*100:.1f}%). Maintain healthy lifestyle and annual screening.")

# --- Model Evaluation Section ---
with st.expander("📊 Show Model Evaluation"):
    st.write(f"### Evaluation Metrics for {model_choice}")
    # Needs matplotlib installed
    st.dataframe(metrics_df.style.format("{:.2f}").background_gradient(cmap="Blues"))

    fig = px.bar(
        metrics_df.reset_index().melt(id_vars=["index"], var_name="Metric", value_name="Score"),
        x="Metric", y="Score", color="index",
        barmode="group", title="Model Performance Comparison",
        color_discrete_sequence=px.colors.sequential.Blues
    )
    fig.update_layout(xaxis_title="Metric", yaxis_title="Score", legend_title="Model")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("<hr><p style='text-align: center; color: grey;'>© 2025 Retinopathy Prediction System</p>", unsafe_allow_html=True)
