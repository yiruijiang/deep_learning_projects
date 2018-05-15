import keras
from keras.models import Sequential
from keras import layers as L
import data_sample

def create_generator_model():

    model = Sequential()

    model.add(L.InputLayer(input_shape = [data_sample.CODE_SIZE], name = "noise"))

    model.add(L.Dense(8 * 8 * 10, activation = 'elu'))

    model.add(L.Reshape([8,8,10]))

    model.add(L.Deconv2D(filters = 64, kernel_size = [5,5], activation = 'elu'))

    model.add(L.Deconv2D(filters = 64, kernel_size = [5,5], activation = 'elu'))

    model.add(L.UpSampling2D(size = [2,2]))

    model.add(L.Deconv2D(filters = 32, kernel_size = [3,3], activation = 'elu'))

    model.add(L.Deconv2D(filters = 32, kernel_size = [3,3], activation = 'elu'))

    model.add(L.Deconv2D(filters = 32, kernel_size = [3,3], activation = 'elu'))

    model.add(L.Conv2D(filters = 3, kernel_size = [3,3], activation = None))

    return model



