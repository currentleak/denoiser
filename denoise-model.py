# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 18:08:08 2024

@author: Kevin.Cotton
"""
import numpy as np
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
#autoencoder.save('autoencoder_denoiser.h5')
autoencoder.save('autoencoder_denoiser.keras')

