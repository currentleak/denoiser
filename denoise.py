# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 18:08:08 2024

@author: Kevin.Cotton
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense
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
noisy_data_directory = './beam-data-noisy/'
clean_data_directory = './beam-data-clean/'
# Load the data from text files in the directory
noisy_data = load_data_from_directory(noisy_data_directory)
clean_data = load_data_from_directory(clean_data_directory)
# Function to read data from multiple text files
# def load_data_from_files(file_prefix, num_files):
#     data = []
#     for i in range(1, num_files + 1):
#         file_name = f'{file_prefix}{i}.txt'
#         file_data = np.loadtxt(file_name)
#         data.append(file_data)
#     return np.concatenate(data)
# Number of files
num_files = 100
# Load the data from text files
# noisy_data = load_data_from_files('./beam-data-noisy/donnee', num_files)
# clean_data = load_data_from_files('./beam-data-clean/donnee', num_files)


# Ensure the data is 2D and has the correct shape
# Assuming each file contains 20000 lines, you can reshape it to (1000, 20) for example
segment_length = 6000 # Length of each segment
noisy_data = noisy_data.reshape(-1, segment_length)
clean_data = clean_data.reshape(-1, segment_length)
# Ensure the data has the correct shape
assert noisy_data.shape == clean_data.shape, "Noisy and clean data must have the same shape"


# Define the autoencoder model
input_dim = noisy_data.shape[1]
encoding_dim = 10

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the model
autoencoder.fit(noisy_data, clean_data, epochs=50, batch_size=256, shuffle=True)

# Save the model
autoencoder.save('autoencoder_denoiser.h5')


# TEST DU MODEL

# Load the trained model
#autoencoder = Model('autoencoder_denoiser.h5')

# Predict (de-noise) using the model
reconstructed_data = autoencoder.predict(noisy_data)


# Plot the data for comparison
def plot_oscilloscope_view(noisy, clean, reconstructed, index):
    time = np.arange(noisy.shape[1]) * 8e-9 # Assuming time is represented by the index of the data points

    # Determine y-axis limits
    y_min = min(np.min(noisy), np.min(clean), np.min(reconstructed))
    y_max = max(np.max(noisy), np.max(clean), np.max(reconstructed))
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Noisy Data")
    plt.plot(time, noisy[index], label='Noisy')
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.ylim([y_min, y_max])
    
    plt.subplot(1, 3, 2)
    plt.title("Clean Data")
    plt.plot(time, clean[index], label='Clean', color='green')
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.ylim([y_min, y_max])
    
    plt.subplot(1, 3, 3)
    plt.title("Reconstructed Data")
    plt.plot(time, reconstructed[index], label='Reconstructed', color='red')
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.ylim([y_min, y_max])
    
    plt.tight_layout()
    plt.show()

# Visualize the first data point
plot_oscilloscope_view(noisy_data, clean_data, reconstructed_data, index=0)

# Visualize more data points if needed
for i in range(1, 10):  # Change the range as needed
    plot_oscilloscope_view(noisy_data, clean_data, reconstructed_data, i)
