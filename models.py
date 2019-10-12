import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

def get_model_1(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model

def get_func_model_1(input_shape, num_classes):
    input_layer = Input(input_shape)
    conv1 = Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv1)
    max_pool = MaxPooling2D(pool_size=(2, 2))(conv2)
    flat = Flatten()(max_pool)
    dense1 = Dense(10, activation='relu')(flat)
    output = Dense(num_classes, activation='softmax')(dense1)
    model = Model(inputs=input_layer, outputs=output)
    return model


def get_model_2(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25)) # Pipeline 4, 5: not commented

    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5)) # Pipeline 4, 5: not commented
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model
