import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.layers import Input, regularizers
from keras.layers import BatchNormalization
from keras.layers import Input, concatenate
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K


#Here is the model initialization
def define_split_model_plus(split, l1 = 0, l2 = 0):

#    activation = 'softplus' #this performs second  best after sigmoid
    activation = 'sigmoid'
    K.clear_session()

    outputs = []
    inputs = []


    for i in range(split.n_filters):
        for j in range(split.n_chunks):
            name = 'chunk' + str(j) + '_filter' + str(i)
            shape = (split.data_shape())[j]
            input = Input(shape=shape, name='input_' + name)
            inputs.append(input)
            output = Dense(8, kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                           bias_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                           activation = activation, name='dense_' + name)(input)
            outputs.append(output)

    first_layer = Model(inputs=inputs, outputs=outputs)
    combined = concatenate(first_layer.outputs)
    #dropout1 = Dropout(0.0)(combined)
    final_layers = Dense(256, kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                         bias_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                         activation = activation, name="Final_hidden_layer1")(combined)
    #final_layers = Dropout(0.3)(final_layers)
    final_layers = Dense(128, activation = activation,
                         kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                         bias_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                         name="Final_hidden_layer2")(final_layers)
    #final_layers = Dropout(0.3)(final_layers)
    final_layers = Dense(10, activation = 'linear', name="Output_layer")(final_layers)

    model = Model(inputs = inputs, outputs = final_layers)

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()


    return model



