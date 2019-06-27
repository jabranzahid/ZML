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
def define_pca_model(ncomp, l1 = 0, l2 = 0):

    ACTIVATION = 'sigmoid'
    model = Sequential()

    #Batch normaliztion produces terrible results. Do not use!
    model.add(Dense(20, activation=ACTIVATION, input_dim=ncomp,
                    kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                    bias_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
#    model.add(Dense(32, activation=ACTIVATION,
#                    kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
#                    bias_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
    model.add(Dense(15, activation=ACTIVATION,
                    kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                    bias_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
    model.add(Dense(10, activation="linear"))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()


    return model



