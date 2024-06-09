# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 18:08:08 2024

@author: Kevin.Cotton
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

# Generate synthetic data (use the same seed for reproducibility)
np.random.seed(42)
noisy_data = np.random.normal(0, 1, (10, 20))  # Use smaller data for visualization
clean_data = np.random.normal(0, 0.5, (10, 20))

# Load the trained model
autoencoder = load_model('autoencoder_denoiser.h5')

# Predict (de-noise) using the model
reconstructed_data = autoencoder.predict(noisy_data)

# Plot the data for comparison
def plot_data(noisy, clean, reconstructed, index):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Noisy Data")
    plt.imshow(noisy[index].reshape(1, -1), aspect='auto', cmap='viridis')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.title("Clean Data")
    plt.imshow(clean[index].reshape(1, -1), aspect='auto', cmap='viridis')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.title("Reconstructed Data")
    plt.imshow(reconstructed[index].reshape(1, -1), aspect='auto', cmap='viridis')
    plt.colorbar()
    
    plt.show()

# Visualize the first data point
plot_data(noisy_data, clean_data, reconstructed_data, index=0)

# Visualize more data points if needed
for i in range(1, 3):  # Change the range as needed
    plot_data(noisy_data, clean_data, reconstructed_data, i)