import argparse
import pickle

import numpy as np
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img


def add_noise(image):
    """Add gaussian noise to image"""
    image_a = img_to_array(image)
    row, col, ch= image_a.shape
    mean = 0
    var = 256//4
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    image_a = image_a + gauss
    image = array_to_img(image_a)
    return image


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
    parser.add_argument('--noise', dest='noise', action='store_true', help="Add noise to training images")
    parser.add_argument('--no-noise', dest='noise', action='store_false', help="Do not add noise to training images")
    parser.set_defaults(noise=False)
    args = parser.parse_args()

    model = get_model(50)

    if args.noise:
        train_datagen = ImageDataGenerator(preprocessing_function=add_noise)
        print('Using noise')
    else:
        train_datagen = ImageDataGenerator()
        print('Without noise')

    train_generator = train_datagen.flow_from_directory(
        args.train,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical')

    test_datagen = ImageDataGenerator()
    test_generator = test_datagen.flow_from_directory(
        args.test,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical')

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['categorical_accuracy', 'top_k_categorical_accuracy'])
    history = model.fit_generator(generator=train_generator, steps_per_epoch=64, epochs=256,
                    validation_data=test_generator, validation_steps=5)

    with open(args.history, 'wb') as f:
        pickle.dump(history.history, f)

    model.save(args.model)
