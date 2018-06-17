import argparse

from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
import pickle


def BirdsResNet50(input_shape, classes_no, fine_tune=False):
    """
    BirdsResNet50 is a modified ResNet50 network with properly adjusted number of units
    in the last fully connected layer.

    :param input_shape: The shape of the input image.
    :param classes_no: The number of classes.
    :param fine_tune: If true, freezes all layers but the last one already added.
    :return: Neural network model.
    """
    base_model = VGG16(weights='imagenet', include_top=True, input_shape=input_shape)

    x = base_model.get_layer('fc2').output
    predictions = Dense(classes_no, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    if fine_tune:
        for layer in base_model.layers:
            layer.trainable = False

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune VGG16')
    parser.add_argument("-train", help="Path to train data")
    parser.add_argument("-test", help="Path to test data")
    parser.add_argument("-history", help="File where to store history")
    parser.add_argument("-model", help="File where to store model")
    args = parser.parse_args()

    train_datagen = ImageDataGenerator()
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

    model = BirdsResNet50(input_shape=(224, 224, 3), classes_no=50,
                          fine_tune=True)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['categorical_accuracy', 'top_k_categorical_accuracy'])
    history = model.fit_generator(generator=train_generator, steps_per_epoch=64, epochs=256,
                    validation_data=test_generator, validation_steps=5)

    with open(args.history, 'wb') as f:
        pickle.dump(history.history, f)

    model.save(args.model)
