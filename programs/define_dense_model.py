from astropy.io import fits
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.layers import Input, regularizers
from keras.layers import BatchNormalization
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plot
import math as math
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import load_model



#Here is the model initialization
def define_dense_model():

    ACTIVATION = 'relu'
    model = Sequential()

    #Batch normaliztion produces terrible results. Do not use!

    model.add(Dense(256, input_shape=(2482,), activation=ACTIVATION))
    model.add(Dense(128, activation=ACTIVATION))
    #model.add(Dropout(0,1))
    model.add(Dense(64, activation=ACTIVATION))
    model.add(Dense(11, activation="linear"))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()


    return model



