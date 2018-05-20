from random import seed

import numpy as np
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.models import Sequential

# root_directory = '../data/SET_A'
from src.data_management import BirdsDataManager, BirdsDataGenerator

root_directory = '../data/BBPP'


def get_model(num_classes):
    model = Sequential()

    model.add(Conv2D(filters=50, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(Conv2D(filters=50, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=50, kernel_size=(3, 3), activation='relu'))

    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=50, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=50, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=50, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=50, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=50, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=50, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


if __name__ == '__main__':
    np.random.seed(4)
    seed(5)

    birds_data_manager = BirdsDataManager(root_directory)

    train_data, test_data = birds_data_manager.get_train_test_dataset(train_size=0.8, random_state=143)

    train_generator = BirdsDataGenerator(train_data, batch_size=30)
    test_generator = BirdsDataGenerator(test_data, batch_size=30)

    model = get_model(50)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['categorical_accuracy', 'top_k_categorical_accuracy'])
    model.fit_generator(generator=train_generator.generate(), steps_per_epoch=20, epochs=20,
                    validation_data=test_generator.generate(), validation_steps=5)

    model.save('birdsmodel_our_bounding.h5')
