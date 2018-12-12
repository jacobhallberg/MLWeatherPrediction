import tensorflow.keras.layers as layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K 

def euclidean_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1, keepdims=True))

def weather_classifier(num_classes, input_shape = (500, 320, 5)):
    model = Sequential()

    model.add(layers.Conv2D(64, 10, strides=2, input_shape=input_shape, data_format="channels_last"))
    # model.add(layers.BatchNormalization())    
    model.add(layers.Activation("relu"))

    model.add(layers.Conv2D(32, 10, strides=2, data_format="channels_last"))
    # model.add(layers.BatchNormalization())    
    model.add(layers.Activation("relu"))

    model.add(layers.Conv2D(num_classes, 10, strides=2, data_format="channels_last"))
    # model.add(layers.BatchNormalization())    
    model.add(layers.Activation("relu"))

    model.add(layers.GlobalAveragePooling2D())
    # model.add(layers.Flatten())
    # model.add(layers.Dense(num_classes))
    model.add(layers.Activation("softmax"))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3),metrics=['accuracy'])

    return model

def ConvLSTM_predictor(input_shape = (10, 500, 320, 5)):
    model = Sequential()

    #Embed down to latent space.
    # ConvLSTM2D (batch, steps, width, height, features)
    model.add(layers.ConvLSTM2D(128, 5, return_sequences=True, input_shape=input_shape))
    model.add(layers.BatchNormalization())    
    model.add(layers.Activation("relu"))

    model.add(layers.ConvLSTM2D(64, 5, return_sequences=True))
    model.add(layers.BatchNormalization())    
    model.add(layers.Activation("relu"))
    
    # Latent Space
    model.add(layers.ConvLSTM2D(32, 5, return_sequences=False))
    model.add(layers.BatchNormalization())    
    model.add(layers.Activation("relu"))

    # Rebuild latent embedding into future prediction.
    model.add(layers.Conv2DTranspose(64, 5))
    model.add(layers.BatchNormalization())    
    model.add(layers.Activation("relu"))

    model.add(layers.Conv2DTranspose(96, 5))
    model.add(layers.BatchNormalization())    
    model.add(layers.Activation("relu"))
    
    model.add(layers.Conv2DTranspose(128, 5))
    model.add(layers.BatchNormalization())    
    model.add(layers.Activation("relu"))

    # Rebuilt future 
    model.add(layers.Conv2DTranspose(5, 5))

    model.compile(loss=euclidean_loss, optimizer=Adam(lr=1e-3),metrics=['accuracy'])