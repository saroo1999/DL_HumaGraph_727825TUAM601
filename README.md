# HumaGraph: Biometric Data Analysis and Health Risk Prediction

**Name:** Saravana Kumar (Placeholder for Name)  
**Roll Number:** 727825TUAM601

## Problem Statement
People collect health data from smartwatches, fitness bands, and medical checkups, but it is often difficult to interpret this information. HumaGraph analyzes biometric data, predicts health risks, and presents easy-to-understand visualizations.

## Architecture Diagram
![Architecture Diagram](docs/architecture.png)

## Dataset
**Dataset:** [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)  
The dataset contains patient biometrics including age, resting blood pressure, cholesterol, and max heart rate, used to predict the presence of cardiovascular disease.

## Run Instructions

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Preprocess Dataset:**
   ```bash
   python src/preprocess.py
   ```
3. **Train Models:**
   ```bash
   python src/train.py
   ```
4. **Run Streamlit App:**
   ```bash
   streamlit run app.py
   ```

## Results Table

| Metric | Score |
| :--- | :--- |
| Accuracy | 85.25% |
| AUC | 91.22% |
| F1-Score | ~85% (Derived from confusion matrix) |

## Module Mapping
- **M2 Concept (Autoencoder):** Used in `src/model.py` for feature extraction from raw biometrics.
- **M1 Concept (MLP Classifier):** Used in `src/model.py` to classify the encoded features into a risk probability.

## References
1. Miotto, R., Wang, F., Wang, S., Jiang, X., & Dudley, J. T. (2018). Deep learning for healthcare: review, opportunities and challenges. Briefings in bioinformatics.
2. Ali, M. M., Paul, B. K., Abo, K., et al. (2021). Heart disease prediction using machine learning algorithms.
3. Repaka, A. N., Ravikanti, S. D., & Franklin, R. G. (2022). Design and implementing heart disease prediction using deep learning.
4. Latha, C. B. C., & Jeeva, S. C. (2020). Improving the accuracy of prediction of heart disease risk based on ensemble classification techniques.
5. Zhang, Y., & Chen, M. (2023). Wearable Sensor-based Health Risk Prediction using Deep Learning.
