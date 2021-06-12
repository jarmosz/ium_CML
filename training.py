import sys

import pandas as pd
import numpy as np
import tensorflow as tf
import os.path

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

arguments = sys.argv[1:]

verbose = int(arguments[0])
epochs = int(arguments[1])

pd.set_option("display.max_columns", None)

# Wczytanie danych
train_data = pd.read_csv("./MoviesOnStreamingPlatforms_updated.train")

# Stworzenie modelu 
columns_to_use = ['Year', 'Runtime', 'Netflix']
train_X = tf.convert_to_tensor(train_data[columns_to_use])
train_Y = tf.convert_to_tensor(train_data[["IMDb"]])

normalizer = preprocessing.Normalization(input_shape=[3,])
normalizer.adapt(train_X)

model = keras.Sequential([
    keras.Input(shape=(len(columns_to_use),)),
    normalizer,
    layers.Dense(30, activation='relu'),
    layers.Dense(10, activation='relu'),
    layers.Dense(25, activation='relu'),
    layers.Dense(1)
])

model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001),
                metrics=[tf.keras.metrics.RootMeanSquaredError()])

model.fit(train_X, train_Y, verbose=verbose, epochs=epochs)

model.save('linear_regression.h5')