import keras.layers as layers
from keras.models import Sequential
from keras.optimizers import Adam
from keras import backend as K

def CLSTM_classifier(timesteps, features):
    model = Sequential()
    model.add(layers.Conv1D(filters=128, kernel_size=3, strides=1, input_shape=(timesteps, features)))
    model.add(layers.Activation('relu'))
    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.Activation('relu'))
    model.add(layers.LSTM(128))
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(256))
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(256))
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(11, activation='softmax', name="OutputLayer"))
    
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3),metrics=['accuracy'])
    return model

def LSTM_classifier(timesteps, features):
    model = Sequential()
    model.add(layers.Masking(mask_value=0., input_shape=(timesteps, features)))
    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.Activation('elu'))
    model.add(layers.LSTM(128))
    model.add(layers.Activation('elu'))
    model.add(layers.Dense(256))
    model.add(layers.Activation('elu'))
    model.add(layers.Dense(128))
    model.add(layers.Activation('elu'))
    model.add(layers.Dense(11, activation='softmax', name="OutputLayer"))
    
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3),metrics=['accuracy'])
    return model

def weather_classifier(num_classes, input_shape = (500, 320, 5)):
    model = Sequential()

    model.add(layers.Conv2D(64, 10, strides=2, input_shape=input_shape, data_format="channels_last"))
    model.add(layers.BatchNormalization())    
    model.add(layers.Activation("relu"))

    model.add(layers.Conv2D(32, 10, strides=2, data_format="channels_last"))
    model.add(layers.BatchNormalization())    
    model.add(layers.Activation("elu"))

    model.add(layers.Conv2D(32, 10, strides=2, data_format="channels_last"))
    model.add(layers.BatchNormalization())    
    model.add(layers.Activation("elu"))

    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dense(50))
    model.add(layers.Dense(num_classes))
    model.add(layers.Activation("softmax"))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3),metrics=['accuracy'])

    return model


def euclidean_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1, keepdims=True))


def ConvLSTM_predictor1(num_classes=1, input_shape = (10, 500, 320, 5)):
    model = Sequential()

    #Embed down to latent space.
    # ConvLSTM2D (batch, steps, width, height, features)
    model.add(layers.ConvLSTM2D(40, 10, return_sequences=True, input_shape=input_shape))
    model.add(layers.BatchNormalization())    
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.ConvLSTM2D(30, 5, strides=2, return_sequences=True))
    model.add(layers.BatchNormalization())    
    model.add(layers.LeakyReLU(alpha=0.2))
    
    # Latent Space
    model.add(layers.ConvLSTM2D(20, 5, return_sequences=False))
    model.add(layers.BatchNormalization())    
    model.add(layers.LeakyReLU(alpha=0.2))

    # Rebuild latent embedding into future prediction.
    model.add(layers.Conv2DTranspose(40, 5))
    model.add(layers.BatchNormalization())    
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(30, 10, strides=2))
    model.add(layers.BatchNormalization())    
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Conv2DTranspose(20, 30))
    model.add(layers.BatchNormalization())    
    model.add(layers.LeakyReLU(alpha=0.2))

    # Rebuilt future 
    model.add(layers.Conv2DTranspose(num_classes, 5))

    model.compile(loss=euclidean_loss, optimizer=Adam(lr=1e-3))
    
    return model


def ConvLSTM_predictor2(num_classes=1, input_shape = (10, 500, 320, 5)):
    model = Sequential()

    # ConvLSTM2D (batch, steps, width, height, features)
    model.add(layers.ConvLSTM2D(30, 10, return_sequences=True, input_shape=input_shape))
    model.add(layers.Activation("relu"))

    model.add(layers.ConvLSTM2D(num_classes, 5, return_sequences=False))
    model.add(layers.Activation("relu"))

    model.compile(loss=euclidean_loss, optimizer=Adam(lr=1e-3))
    
    return model
