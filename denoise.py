# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 18:08:08 2024

@author: Kevin.Cotton
"""
import numpy as np
import matplotlib.pyplot as plt
import keras
#from keras.models import Model
#from keras.layers import Input, Dense
import os

def load_data_from_directory(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            file_data = np.loadtxt(file_path)
            data.append(file_data)
    return np.concatenate(data)

# Directory containing the text files
real_data_directory = './beam-data-real/'
# Load the data from text files in the directory
real_data = load_data_from_directory(real_data_directory)


# Ensure the data is 2D and has the correct shape
# Assuming each file contains 20000 lines, you can reshape it to (1000, 20) for example
segment_length = 6000 # Length of each segment
real_data = real_data.reshape(-1, segment_length)


# Load the trained model
autoencoder = keras.models.load_model("autoencoder_denoiser.keras")
#autoencoder = Model('autoencoder_denoiser.keras')

# Predict (de-noise) using the model
reconstructed_data = autoencoder.predict(real_data)


# Plot the data for comparison
def plot_oscilloscope_view(real, reconstructed, index):
    time = np.arange(real.shape[1]) * 8e-9 # Assuming time is represented by the index of the data points

    # Determine y-axis limits
    y_min = min(np.min(real), np.min(reconstructed))
    y_max = max(np.max(real), np.max(reconstructed))
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Real Data")
    plt.plot(time, real[index], label='Noisy')
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.ylim([y_min, y_max])
    
    plt.subplot(1, 2, 2)
    plt.title("Reconstructed Data")
    plt.plot(time, reconstructed[index], label='Reconstructed', color='red')
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.ylim([y_min, y_max])
    
    plt.tight_layout()
    plt.show()

# Visualize the first data point
plot_oscilloscope_view(real_data, reconstructed_data, index=0)

# Visualize more data points if needed
for i in range(1, 4):  # Change the range as needed
    plot_oscilloscope_view(real_data, reconstructed_data, i)
