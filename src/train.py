import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from model import build_autoencoder, build_classifier

def train_and_evaluate():
    # Load data
    print("Loading datasets...")
    X_train = pd.read_csv('datasets/X_train.csv')
    X_test = pd.read_csv('datasets/X_test.csv')
    y_train = pd.read_csv('datasets/y_train.csv')
    y_test = pd.read_csv('datasets/y_test.csv')
    
    input_dim = X_train.shape[1]
    encoding_dim = 8
    
    # 1. Train Autoencoder
    print("Training Autoencoder...")
    autoencoder, encoder = build_autoencoder(input_dim, encoding_dim)
    
    history_ae = autoencoder.fit(
        X_train, X_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    # Extract features using Encoder
    print("Extracting features with Encoder...")
    encoded_X_train = encoder.predict(X_train)
    encoded_X_test = encoder.predict(X_test)
    
    # 2. Train Classifier
    print("Training Classifier...")
    classifier = build_classifier(encoding_dim)
    
    history_clf = classifier.fit(
        encoded_X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(encoded_X_test, y_test),
        verbose=1
    )
    
    # Evaluate
    loss, accuracy, auc = classifier.evaluate(encoded_X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}, Test AUC: {auc:.4f}")
    
    # Save models
    os.makedirs('models', exist_ok=True)
    encoder.save('models/encoder.h5')
    classifier.save('models/best_model.h5')
    print("Models saved successfully.")
    
    # Generate Plots
    os.makedirs('assets', exist_ok=True)
    
    # Plot accuracy and loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(history_clf.history['accuracy'], label='Train Accuracy')
    ax1.plot(history_clf.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Classifier Accuracy')
    ax1.legend()
    
    ax2.plot(history_clf.history['loss'], label='Train Loss')
    ax2.plot(history_clf.history['val_loss'], label='Val Loss')
    ax2.set_title('Classifier Loss')
    ax2.legend()
    plt.savefig('assets/training_curves.png')
    
    # Confusion Matrix
    y_pred = (classifier.predict(encoded_X_test) > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('assets/confusion_matrix.png')
    
    print("Training complete. Plots saved to assets/ directory.")

if __name__ == "__main__":
    train_and_evaluate()
