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
import pandas as pd
import matplotlib.pyplot as plot
import math as math
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import load_model


DATA_FILE_PATH ='/Users/jabran/ml/metallicity/data/'

def get_training_data(norm = False, SNR = 0):
#read in training data
#these files were produced in IDL using wrapper_fsps_sfh_z_tabular_ascii.pro
# and make_evol_fsps_model_str.pro. I fiddled with the programs,
#one set is normalized the other is not using /no_norm keyword

    if norm == False:
        FILE = "fsps_evol_models_no_norm.fits"
    else:
        FILE = "fsps_evol_models_norm.fits"

    FSPS_FILE = DATA_FILE_PATH + FILE
    hdul = fits.open(FSPS_FILE)
    data = hdul[1].data
    flux = data.field(1)

    if SNR != 0:
        noise = np.random.normal(size=(1030, 3249))*(flux/SNR)
    else:
        noise = 0

    flux = flux + noise - 1
    z_solar =0.0142
    lwz = data['LWZ']/z_solar
    lwa = data['LWA']

    #this came from mask_emission_sdss_andrews.pro
    MASK_FILE = "/Users/jabran/ml/metallicity/data/emission_line_mask.txt"
    mask = np.loadtxt(MASK_FILE)
    index = np.where(mask == 0)
    flux_mask = (flux[:,index].reshape(1030, 2482))
    index_zlo = np.where(lwz < 3.1)
    nsel = len(index_zlo[0])
    flux_mask = (flux_mask[index_zlo,:].reshape(nsel, 2482))
    features = np.expand_dims(flux_mask, axis=2)
    n_flux = len(flux_mask[0,:])
    n_spec = len(flux_mask)
    labels = np.stack((lwz[index_zlo], lwa[index_zlo]), axis=1)

    #randomly reshuffle before feeding CNN
    np.random.seed(4)
    np.random.shuffle(features)
    np.random.seed(4)
    np.random.shuffle(labels)

    return features, labels




def get_test_data(norm = False, high_mass = True, sfr_sort = False):


    if sfr_sort:
        if norm:
            FILE = "sdss_sort_stack_data_norm.fits"
        else:
            FILE = "sdss_sort_stack_data_no_norm.fits"
    else:
        if norm:
            FILE = "sdss_stack_data_norm.fits"
        else:
            FILE = "sdss_stack_data_no_norm.fits"



    STACK_FILE = DATA_FILE_PATH + FILE

    hdul = fits.open(STACK_FILE)
    data = hdul[1].data
    flux_test = data.field(1) - 1
    mass = data['MASS']

    #this came from mask_emission_sdss_andrews.pro
    MASK_FILE = "/Users/jabran/ml/metallicity/data/emission_line_mask.txt"
    mask = np.loadtxt(MASK_FILE)
    index = np.where(mask == 0)

    if sfr_sort:
        flux_test_mask = (flux_test[:,index].reshape(170, 2482))
        flux_test_mask = np.expand_dims(flux_test_mask, axis=2)
    else:
        flux_test_mask = (flux_test[:,index].reshape(34, 2482))
        flux_test_mask = np.expand_dims(flux_test_mask, axis=2)


    if high_mass:
        if sfr_sort:
            ind = 45
        else:
            ind = 9
        flux_test_mask = flux_test_mask[ind:,:,:]
        mass = mass[ind:]


    return flux_test_mask, mass




#Here is the model initialization
def define_cnn_model():

    model = Sequential()

    #Batch normaliztion produces terrible results. Do not use!

    model.add(Conv1D(16, 2482, input_shape=(2482, 1), activation="relu"))
#    model.add(Conv1D(4, 5, input_shape=(2482, 1), activation="relu"))
#    model.add(Conv1D(8, 5, activation="relu"))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0,1))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(2, activation="linear"))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()


    return model



