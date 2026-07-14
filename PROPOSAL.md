# Project Proposal: HumaGraph

**Title:** HumaGraph: Biometric Data Analysis and Health Risk Prediction  
**Name & Roll Number:** 727825TUAM601  
**Course:** 23ADC04 DEEP LEARNING

## Objective
The main objective of this project is to develop a deep learning pipeline that analyzes biometric data (e.g., heart rate, blood pressure, cholesterol levels) to predict the risk of cardiovascular diseases. The system aims to provide an easy-to-understand visualization of these risks to help individuals interpret their health data effectively.

## Dataset Source
**Dataset:** UCI Heart Disease Dataset  
**Source:** UCI Machine Learning Repository (also available via Kaggle and `ucimlrepo` Python package)  
**Description:** The dataset contains 14 attributes including age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting electrocardiographic results, maximum heart rate achieved, exercise induced angina, and others, targeting the presence of heart disease in the patient.

## Architecture Overview
The project will implement a hybrid deep learning model comprising:
1. **Module 2 Concept (Autoencoder):** An Autoencoder will be utilized for feature extraction and dimensionality reduction on the continuous biometric features, effectively learning a robust representation of the health profile.
2. **Module 1 Concept (MLP):** The latent representation from the Autoencoder, concatenated with categorical features, will be fed into a Multi-Layer Perceptron (MLP) which will act as the final classifier to output the probability of heart disease risk.

## Expected Outcome
- A functional deep learning model capable of predicting cardiovascular disease risk based on biometric parameters.
- A user-friendly Streamlit web application that accepts user biometrics, predicts the risk, and displays easy-to-understand visualizations.
- Comprehensive evaluation metrics including Accuracy, F1-score, Precision, Recall, Confusion Matrix, and ROC-AUC.
