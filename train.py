'''Example script for training fusion network.
Data is not a part of this repository, so the 
script can not be run out-of-the-box.
'''

import keras
from keras.callbacks import ReduceLROnPlateau
from model import ASPP_model_extdata
import numpy as np

batch_size = 32
epochs = 80

#Load data
X_s1_train = # Load data of format, e.g (n, 300, 300, 2) for sentinel-1
X_am_train = # Load data of format, e.g (n, 6, 6, 14) for AMSR-2
Y_sic_train = # Load data of format, e.g (n, 300, 300, 1) for SIC (sea ice concentration)

X_s1_valid = # Load data of format, e.g (n, 300, 300, 2) for sentinel-1
X_am_valid = # Load data of format, e.g (n, 6, 6, 14) for sentinel-1
Y_sic_valid = # Load data of format, e.g (n, 300, 300, 1) for SIC (sea ice concentration)

print('x_train shape:', X_s1_train.shape)
print(X_s1_train.shape[0], 'train samples')
print(X_s1_train.shape[0], 'test samples')

# The input of the CNN model is a list of Sentinel-1 and AMSR2 pairs.
model = ASPP_model_extdata([X_s1_train.shape[1:], X_am_train.shape[1:]])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

def generator(X_s1, X_am, Y_sic, batch_size):
    #Shuffle order of batches
    arr = np.random.shuffle(np.arange(int(X_s1.shape[0]//batch_size)))

    #loop over batches
    for i in arr:
        X_s1_batch = X_s1[i*batch_size:(i+1)*batch_size]
        X_am_batch = X_am[i*batch_size:(i+1)*batch_size]
        Y_sic_batch = Y_sic[i*batch_size:(i+1)*batch_size]

        yield([X_s1_batch, X_am_batch], Y_sic_batch)

history = model.fit_generator(generator(X_s1_train, X_am_train, Y_sic_train, batch_size),
                              steps_per_epoch=(X_s1_train.shape[0]//batch_size).astype(int),
                              callbacks=[ReduceLROnPlateau(monitor='loss', 
                                                           factor=0.2,
                                                           patience=5, 
                                                           min_lr=0.0001,
                                                           min_delta=0.001)],
                              validation_data=generator(X_s1_valid, X_am_valid, Y_sic_valid, batch_size), 
                              validation_steps=(X_s1_valid.shape[0]//batch_size).astype(int),
                              epochs=epochs, 
                              verbose=1)

print('Test loss:', history.history['val_loss'][-1])
print('Test accuracy:', history.history['val_acc'][-1])