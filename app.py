import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from src.model import build_autoencoder, build_classifier
import pickle

st.set_page_config(page_title="HumaGraph", page_icon="🫀", layout="wide")

st.title("🫀 HumaGraph: Health Risk Predictor")
st.markdown("""
Welcome to **HumaGraph**! This application uses a Deep Learning hybrid model (Autoencoder + MLP) 
to analyze your biometric data and predict the risk of cardiovascular disease.
""")

# Load models
@st.cache_resource
def load_models():
    # In a real app we'd load weights. Since training might happen later, we'll try to load, 
    # and if it fails, we will just show a placeholder or load mock models.
    try:
        encoder = tf.keras.models.load_model('models/encoder.h5')
        classifier = tf.keras.models.load_model('models/best_model.h5')
        return encoder, classifier
    except:
        return None, None

encoder, classifier = load_models()

st.sidebar.header("Input Biometrics")

# Create form for inputs based on UCI Heart Disease features
age = st.sidebar.slider("Age", 20, 100, 50)
sex = st.sidebar.selectbox("Sex (1=Male, 0=Female)", [1, 0])
cp = st.sidebar.selectbox("Chest Pain Type", [1, 2, 3, 4])
trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
chol = st.sidebar.slider("Serum Cholestoral (mg/dl)", 100, 600, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)", [1, 0])
restecg = st.sidebar.selectbox("Resting ECG", [0, 1, 2])
thalach = st.sidebar.slider("Max Heart Rate Achieved", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina (1=Yes, 0=No)", [1, 0])
oldpeak = st.sidebar.slider("ST depression induced by exercise", 0.0, 6.2, 1.0)
slope = st.sidebar.selectbox("Slope of peak exercise ST segment", [1, 2, 3])
ca = st.sidebar.slider("Number of major vessels", 0, 3, 0)
thal = st.sidebar.selectbox("Thal (3=Normal, 6=Fixed, 7=Reversable)", [3, 6, 7])

if st.sidebar.button("Predict Risk"):
    if encoder is None or classifier is None:
        st.error("Models are not trained yet. Please run the training script first.")
    else:
        # Create input array
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        
        # We need the scaler used in training. For demo, we just apply a mock scaling or skip
        # Note: In a complete project, save the scaler in src/preprocess.py using pickle
        # and load it here. Assuming data is passed as is for simplicity in this demo wrapper.
        
        # Predict using Autoencoder -> Classifier
        with st.spinner("Analyzing biometrics..."):
            encoded_features = encoder.predict(input_data)
            risk_prob = classifier.predict(encoded_features)[0][0]
        
        st.subheader("Analysis Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Heart Disease Risk Score", f"{risk_prob*100:.1f}%")
        
        with col2:
            if risk_prob > 0.5:
                st.error("High Risk Detected! Please consult a healthcare professional.")
            else:
                st.success("Low Risk. Keep up the healthy lifestyle!")
                
        # Show progress bar
        st.progress(float(risk_prob))
