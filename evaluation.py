import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import os.path

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

# Wczytanie danych
test_data = pd.read_csv("./MoviesOnStreamingPlatforms_updated.test")

columns_to_use = ['Year', 'Runtime', 'Netflix']
test_X = tf.convert_to_tensor(test_data[columns_to_use])
test_Y = tf.convert_to_tensor(test_data[["IMDb"]])

model = tf.keras.models.load_model('linear_regression.h5')

scores = model.evaluate(x=test_X, 
                    y=test_Y)    

with open('rmse.txt', 'w') as file:
    file.write(str(scores[1]) + "\n")
