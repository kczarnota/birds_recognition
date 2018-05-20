from random import seed

import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

from src.data_management import BirdsDataManager, BirdsDataGenerator


def BirdsResNet50(input_shape, classes_no, fine_tune=False):
    """
    BirdsResNet50 is a modified ResNet50 network with properly adjusted number of units
    in the last fully connected layer.

    :param input_shape: The shape of the input image.
    :param classes_no: The number of classes.
    :param fine_tune: If true, freezes all layers but the last one already added.
    :return: Neural network model.
    """
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(classes_no, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    if fine_tune:
        for layer in base_model.layers:
            layer.trainable = False

    return model


#root_directory = '../data/SET_A'
root_directory = '../data/BBPP'

if __name__ == '__main__':
    np.random.seed(4)
    seed(5)

    birds_data_manager = BirdsDataManager(root_directory)

    train_data, test_data = birds_data_manager.get_train_test_dataset(train_size=0.8, random_state=143)

    train_generator = BirdsDataGenerator(train_data)
    test_generator = BirdsDataGenerator(test_data)

    model = BirdsResNet50(input_shape=(224, 224, 3), classes_no=len(birds_data_manager.classes),
                          fine_tune=True)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['categorical_accuracy', 'top_k_categorical_accuracy'])
    model.fit_generator(generator=train_generator.generate(), steps_per_epoch=20, epochs=20,
                    validation_data=test_generator.generate(), validation_steps=5)

    model.save('birdsmodel_bounding.h5')
