from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.models import Model


def get_OCR_model():
    input = Input(shape=(32,32,3))
    
    block1 = Conv2D(32, (5,5), padding="same", activation="relu")(input)
    block1 = Conv2D(32, (5,5), padding="same", activation="relu")(block1)
    block1 = Conv2D(32, (5,5), padding="same", activation="relu")(block1)
    block1 = MaxPooling2D((2,2))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Dropout(0.25)(block1)
    
    block2 = Conv2D(64, (3,3), padding="same", activation="relu")(block1)
    block2 = Conv2D(64, (3,3), padding="same", activation="relu")(block2)
    block2 = Conv2D(64, (3,3), padding="same", activation="relu")(block2)
    block2 = MaxPooling2D((2,2))(block2)
    block2 = BatchNormalization()(block2)
    block2 = Dropout(0.25)(block2)
    
    dense = Flatten()(block2)
    dense = Dense(256, activation="relu")(dense)
    dense = Dense(128, activation="relu")(dense)
    dense = Dropout(0.4)(dense)
    
    output = Dense(28, activation="softmax")(dense)
    
    model = Model(input, output)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    return model