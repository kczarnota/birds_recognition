from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras
import numpy as np
import pandas as pd

PARAMETERS = [[30], [10, 20], [5, 10, 15], [3, 6, 9, 12], [2, 4, 5, 8, 10]]


def load_data(trening_test_ratio=0.8):
    df = pd.read_csv('../BSIF/bsif_features.csv')
    array = df.values
    np.random.shuffle(array)
    training_set_size = int(trening_test_ratio * array.shape[0])
    X_train = array[:training_set_size, :-1]
    y_train = array[:training_set_size, -1:].flatten().astype(dtype=np.int32)
    X_test = array[training_set_size + 1:, :-1]
    y_test = array[training_set_size + 1:, -1:].flatten().astype(dtype=np.int32)
    y_train -= 1
    y_test -= 1
    return X_train, y_train, X_test, y_test


def get_model(input_shape, num_classes, hidden_layers_number, base):
    model = Sequential()
    model.add(Dense(PARAMETERS[hidden_layers_number - 1][0] * base, activation='relu', input_shape=input_shape))
    for i in PARAMETERS[hidden_layers_number - 1][1:]:
        model.add(Dense(i * base, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    print(model.summary())
    return model


def run_experiment():
    batch_size = 120
    num_classes = 50
    epochs = 12

    # input image dimensions
    input_shape = (256,)

    # the data, split between train and test sets
    X_train, y_train, X_test, y_test = load_data()

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = get_model(input_shape, num_classes, 5, 16)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(X_test, y_test))
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    run_experiment()
