import numpy as np
import tensorflow as tf
from tensorflow import keras #framework for defining a neural network

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# compiling the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# providing the data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

# training the neural network
model.fit(xs, ys, epochs=50)