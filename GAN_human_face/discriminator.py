import keras
from keras import layers as L
from keras.models import Sequential

import data_sample
import tensorflow as tf

def create_discriminator_model():

    model = Sequential()

    model.add(L.InputLayer(input_shape = data_sample.IMG_SHAPE))

    model.add(L.Conv2D(filters = 32, kernel_size = [3,3]))

    model.add(L.AveragePooling2D(pool_size = [2,2]))

    model.add(L.Activation(activation = 'elu'))

    model.add(L.Conv2D(filters = 64, kernel_size = [3,3]))

    model.add(L.AveragePooling2D(pool_size = [2,2]))

    model.add(L.Activation(activation = 'elu'))

    model.add(L.Flatten())

    model.add(L.Dense(units = 256, activation = 'tanh'))

    model.add(L.Dense(units = 2, activation = tf.nn.log_softmax))

    return model