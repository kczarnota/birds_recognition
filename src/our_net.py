import argparse
import json

from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.models import Sequential
from tensorflow.python.keras._impl.keras.preprocessing.image import ImageDataGenerator


def get_model(num_classes):
    model = Sequential()

    model.add(Conv2D(filters=50, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(Conv2D(filters=50, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=50, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=50, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=50, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=50, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=50, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=50, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=50, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=50, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train our network')
    parser.add_argument("-train", help="Path to train data")
    parser.add_argument("-test", help="Path to test data")
    parser.add_argument("-history", help="File where to store history")
    parser.add_argument("-model", help="File where to store model")
    args = parser.parse_args()

    model = get_model(50)
    datagen = ImageDataGenerator()

    train_generator = datagen.flow_from_directory(
        args.train,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical')

    test_generator = datagen.flow_from_directory(
        args.test,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical')

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['categorical_accuracy', 'top_k_categorical_accuracy'])
    history = model.fit_generator(generator=train_generator, steps_per_epoch=64, epochs=64,
                    validation_data=test_generator, validation_steps=5)

    with open(args.history, 'w') as f:
        json.dump(history, f)

    model.save(args.model)
