import tensorflow as tf
from tensorflow.keras import layers, models

def build_autoencoder(input_dim, encoding_dim=8):
    """
    Builds an Autoencoder model (Module 2 Concept) for feature extraction.
    """
    # Encoder
    encoder_input = layers.Input(shape=(input_dim,), name="encoder_input")
    x = layers.Dense(16, activation="relu")(encoder_input)
    encoder_output = layers.Dense(encoding_dim, activation="relu", name="encoder_output")(x)
    
    # Decoder
    x = layers.Dense(16, activation="relu")(encoder_output)
    decoder_output = layers.Dense(input_dim, activation="linear", name="decoder_output")(x)
    
    autoencoder = models.Model(encoder_input, decoder_output, name="autoencoder")
    encoder = models.Model(encoder_input, encoder_output, name="encoder")
    
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

def build_classifier(encoding_dim):
    """
    Builds an MLP Classifier (Module 1 Concept) that takes the encoded features.
    """
    classifier_input = layers.Input(shape=(encoding_dim,), name="classifier_input")
    x = layers.Dense(32, activation="relu")(classifier_input)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(16, activation="relu")(x)
    classifier_output = layers.Dense(1, activation="sigmoid", name="classifier_output")(x)
    
    classifier = models.Model(classifier_input, classifier_output, name="classifier")
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    
    return classifier
