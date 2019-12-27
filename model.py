from keras.layers import Conv2D, Input, Concatenate
from keras.layers import AveragePooling2D, Dropout, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
import numpy as np
from keras import backend as K
from keras.engine.topology import Layer

def ASPP_model_extdata(insize):
    inputs = []
    for elem in insize:
        inputs.append(Input(shape=elem))

    x = Conv2D(12, (3, 3), padding='same', activation='relu')(inputs[0])
    x_ = Conv2D(12, (3, 3), padding='same', activation='relu')(x)

    x = BatchNormalization()(x_)
    x = Dropout(0.25)(x)

    x = Conv2D(18, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(18, (3, 3), padding='same', activation='relu')(x)

    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(18, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(18, (3, 3), padding='same', activation='relu')(x)

    x1 = AveragePooling2D(pool_size=(2,2), strides=(1,1), padding="same")(x)
    x2 = AveragePooling2D(pool_size=(4,4), strides=(1,1), padding="same")(x)
    x3 = AveragePooling2D(pool_size=(8,8), strides=(1,1), padding="same")(x)
    x4 = AveragePooling2D(pool_size=(16,16), strides=(1,1), padding="same")(x)

    x1 = Conv2D(24, (3, 3), padding='same',
                  dilation_rate=(2, 2), activation='relu')(x1)    
    x2 = Conv2D(24, (3, 3), padding='same',
                  dilation_rate=(4, 4), activation='relu')(x2)
    x3 = Conv2D(24, (3, 3), padding='same',
                  dilation_rate=(8, 8), activation='relu')(x3)
    x4 = Conv2D(24, (3, 3), padding='same',
                  dilation_rate=(16, 16), activation='relu')(x4)

    temp = []
    for i in range(1,len(insize)):
        temp.append(UpSampling2D(size=(insize[0][0]//insize[i][0],
                                     insize[0][1]//insize[i][1]))(inputs[i]))

    x = Concatenate()([x_, x1, x2, x3, x4]+temp)
    predictions = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=predictions)

    return(model)