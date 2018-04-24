from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PARAMETERS = [[30], [10, 20], [5, 10, 15], [3, 6, 9, 12], [2, 4, 5, 8, 10]]


def load_data(train_ratio=0.9, valid_ratio=0.1):
    if train_ratio + valid_ratio > 1.0:
        raise ValueError()

    df = pd.read_csv('../BSIF/bsifhistnorm_features_gray_cube.csv',  header=None)
    array = df.values
    class_labels = np.unique(array[:,-1].astype(np.int32))
    print("Data loading done")
    print("Loaded {} samples splitted into {} classes".format(array.shape[0], class_labels.shape[0]))

    X_train = np.empty((0, array.shape[1] - 1))
    y_train = np.empty((0,1))
    X_test = np.empty((0, array.shape[1] - 1))
    y_test = np.empty((0,1))
    X_valid = np.empty((0, array.shape[1] - 1))
    y_valid = np.empty((0,1))

    for c in np.nditer(class_labels):
        row_filter = array[:,-1].astype(np.int32) == c
        class_samples = array[row_filter]
        np.random.shuffle(class_samples)

        training_set_size = int(train_ratio * class_samples.shape[0])

        if float(training_set_size) != train_ratio * class_samples.shape[0]:
            training_set_size += 1

        validation_set_size = int(valid_ratio * (class_samples.shape[0]))
        if float(validation_set_size) != valid_ratio * class_samples.shape[0]:
            validation_set_size += 1

        test_set_size = class_samples.shape[0] - validation_set_size - training_set_size

        if test_set_size < 0:
            validation_set_size += test_set_size
            test_set_size = 0

        X_train_class = class_samples[:training_set_size, :-1]
        y_train_class = class_samples[:training_set_size, -1:].astype(dtype=np.int32)

        X_valid_class = class_samples[training_set_size:training_set_size + validation_set_size, :-1]
        y_valid_class = class_samples[training_set_size:training_set_size + validation_set_size, -1:].astype(dtype=np.int32)

        if test_set_size != 0:
            X_test_class = class_samples[-test_set_size:, :-1]
            y_test_class = class_samples[-test_set_size:, -1:].astype(dtype=np.int32)


        X_train = np.append(X_train, X_train_class, axis=0)
        y_train = np.append(y_train, y_train_class, axis=0)

        X_valid = np.append(X_valid, X_valid_class, axis=0)
        y_valid = np.append(y_valid, y_valid_class, axis=0)

        if test_set_size != 0:
            X_test = np.append(X_test, X_test_class, axis=0)
            y_test = np.append(y_test, y_test_class, axis=0)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def get_model(input_shape, num_classes, hidden_layers_number, base):
    model = Sequential()
    model.add(Dense(PARAMETERS[hidden_layers_number - 1][0] * base, activation='relu', input_shape=input_shape))
    for i in PARAMETERS[hidden_layers_number - 1][1:]:
        model.add(Dense(i * base, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    print(model.summary())
    return model


def save_plot(history, num_layers):
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.plot(history.history['top_k_categorical_accuracy'])
    plt.plot(history.history['val_top_k_categorical_accuracy'])
    plt.title('Dokładność (liczba warstw = {})'.format(num_layers))
    plt.ylabel('Dokładność')
    plt.xlabel('Epoka')
    plt.legend(['top 1 train', 'top 1 val', 'top 5 train', 'top 5 val'], loc='upper left')
    plt.savefig('acc_layers_{}.pdf'.format(num_layers))
    plt.close()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss (liczba warstw = {})'.format(num_layers))
    plt.ylabel('Loss')
    plt.xlabel('Epoka')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('loss_layers_{}.pdf'.format(num_layers))
    plt.close()


def run_experiment():
    batch_size = 16
    num_classes = 50
    epochs = 256


    # the data, split between train, validation and test sets
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data()

    input_shape = (X_train.shape[1],)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_valid = keras.utils.to_categorical(y_valid, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    scores = []

    for i in range(1, 6):
        model = get_model(input_shape, num_classes, i, 16)
        model.compile(loss=keras.losses.categorical_crossentropy,
                      #optimizer=keras.optimizers.Adadelta(),
                      #optimizer='rmsprop',
                      optimizer='adam',
                      metrics=['categorical_accuracy', 'top_k_categorical_accuracy'])

        history = model.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  shuffle=True,
                  verbose=1,
                  validation_data=(X_valid, y_valid))

        save_plot(history, i)

        score = model.evaluate(X_valid, y_valid, verbose=0)
        scores.append(score[1])
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    for i in range(len(scores)):
        print('{} layers test accuracy: {}'.format(i + 1, scores[i]))


if __name__ == '__main__':
    run_experiment()
