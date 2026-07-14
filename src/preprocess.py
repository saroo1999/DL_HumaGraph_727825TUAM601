import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

def fetch_and_preprocess_data():
    """
    Fetches the UCI Heart Disease dataset, handles missing values,
    normalizes continuous features, and splits into train/test sets.
    """
    # Fetch dataset
    heart_disease = fetch_ucirepo(id=45)
    
    # Data (as pandas dataframes)
    X = heart_disease.data.features
    y = heart_disease.data.targets
    
    # Binarize target (0: no disease, >0: disease presence)
    y = (y > 0).astype(int)
    
    # Combine to handle missing values together
    df = pd.concat([X, y], axis=1)
    
    # Handle missing values (replace with median for simplicity)
    df.fillna(df.median(), inplace=True)
    
    # Separate back
    X = df.drop(columns=['num'])
    y = df['num']
    
    # Identify continuous and categorical features
    # Note: treating most numerical columns as continuous for scaling
    continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    
    # Scale continuous features
    scaler = StandardScaler()
    X[continuous_features] = scaler.fit_transform(X[continuous_features])
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Ensure datasets directory exists
    os.makedirs('datasets', exist_ok=True)
    
    # Save the processed data
    X_train.to_csv('datasets/X_train.csv', index=False)
    X_test.to_csv('datasets/X_test.csv', index=False)
    y_train.to_csv('datasets/y_train.csv', index=False)
    y_test.to_csv('datasets/y_test.csv', index=False)
    
    print("Preprocessing completed. Data saved to 'datasets/' folder.")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    fetch_and_preprocess_data()
