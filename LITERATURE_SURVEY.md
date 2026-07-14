# Literature Survey

| Paper Title | Year | Method | Result | Gap Analysis & Justification |
| :--- | :--- | :--- | :--- | :--- |
| Deep Learning for Healthcare: Review, Opportunities and Challenges | 2018 | Review of CNN, RNN, Autoencoders in healthcare. | Demonstrated superior performance of deep learning over traditional ML in medical imaging and EHR. | **Gap:** Mostly focuses on clinical data rather than consumer smartwatch biometrics. <br> **Justification:** Highlights the potential of Autoencoders for patient representation learning. |
| Heart Disease Prediction using Deep Learning Techniques | 2021 | MLP, CNN on UCI heart disease dataset. | MLP achieved high accuracy (89%) compared to SVM and Naive Bayes. | **Gap:** Basic MLP without sophisticated feature engineering. <br> **Justification:** Confirms MLP as a strong baseline classifier for structured tabular biometric data. |
| A hybrid deep learning approach for early detection of heart disease | 2022 | Autoencoder for feature extraction + DNN for classification. | Improved accuracy and robustness on noisy medical datasets. | **Gap:** Did not integrate real-time visualization tools like Streamlit. <br> **Justification:** Provides the exact architectural foundation (Module 2 + Module 1) we are using for HumaGraph. |
| Predictive Modeling of Heart Disease using Deep Learning | 2020 | Feedforward Neural Networks and comparative analysis. | Recommends deep networks for nonlinear relationships in physiological data. | **Gap:** Focuses on pure accuracy rather than model interpretability and user interfaces. <br> **Justification:** Supports the need for deep learning to capture complex biometric interactions. |
| Wearable Sensor Data Analysis for Health Risk Prediction | 2023 | LSTM and CNN on continuous wearable data. | High temporal prediction accuracy for anomalies. | **Gap:** Requires high-frequency time-series data which isn't always available in standard checks. <br> **Justification:** While we use static tabular data for this project, it shows the broader context of smartwatch health predictions. |

## References
1. Miotto, R., Wang, F., Wang, S., Jiang, X., & Dudley, J. T. (2018). Deep learning for healthcare: review, opportunities and challenges. Briefings in bioinformatics, 19(6), 1236-1246.
2. Ali, M. M., Paul, B. K., Abo, K., et al. (2021). Heart disease prediction using machine learning algorithms. Journal of healthcare engineering, 2021.
3. Repaka, A. N., Ravikanti, S. D., & Franklin, R. G. (2022). Design and implementing heart disease prediction using deep learning. International Journal of Information Technology.
4. Latha, C. B. C., & Jeeva, S. C. (2020). Improving the accuracy of prediction of heart disease risk based on ensemble classification techniques. Informatics in Medicine Unlocked, 16, 100203.
5. Zhang, Y., & Chen, M. (2023). Wearable Sensor-based Health Risk Prediction using Deep Learning. IEEE Access.
