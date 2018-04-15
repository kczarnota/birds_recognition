from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras
import numpy as np
import pandas as pd

def load_data(trening_test_ratio=0.8):
    df = pd.read_csv('../BSIF/bsif_features.csv')
    array = df.values
    np.random.shuffle(array)
    training_set_size = int(trening_test_ratio * array.shape[0])
    X_trening = array[:training_set_size, :-1]
    y_trening = array[:training_set_size, -1:].flatten().astype(dtype=np.int32)
    X_test = array[training_set_size + 1:, :-1]
    y_test = array[training_set_size + 1:, -1:].flatten().astype(dtype=np.int32)
    return X_trening, y_trening, X_test, y_test



def get_model(input_shape, num_classes):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def run_experiment():
    batch_size = 120
    num_classes = 51
    epochs = 12

    # input image dimensions
    input_shape = (256,)

    # the data, split between train and test sets
    x_train, y_train, x_test, y_test = load_data()

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = get_model(input_shape, num_classes)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    run_experiment()
