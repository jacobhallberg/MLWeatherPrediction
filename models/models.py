import tensorflow.keras.layers as layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def weather_classifier(num_classes, input_shape = (100,100,5)):
    model = Sequential()

    model.add(layers.Conv3D(32, 5, input_shape=input_shape))
    model.add(layers.BatchNormalization())    
    model.add(layers.Activation("relu"))

    model.add(layers.Conv3D(64, 5, input_shape=input_shape))
    model.add(layers.BatchNormalization())    
    model.add(layers.Activation("relu"))

    model.add(layers.Conv3D(128, 5, input_shape=input_shape))
    model.add(layers.BatchNormalization())    
    model.add(layers.Activation("relu"))

    model.add(layers.Flatten())
    model.add(layers.Dense(num_classes))
    model.add(layers.Activation("softmax"))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3),metrics=['accuracy'])

    return model

def RNN_predictor():
    pass


