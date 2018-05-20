from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from glob import glob
from random import shuffle, seed
from PIL import Image
import numpy as np
import os


class BirdsDataManager(object):
    """
    Manages birds dataset.

    """
    def __init__(self, root_directory):
        self.root_directory = root_directory
        self.classes = next(os.walk(root_directory))[1]
        self.label_encoder = preprocessing.LabelEncoder()
        self.label_encoder.fit(self.classes)

    def get_train_test_dataset(self, train_size, random_state):
        train_data = []
        test_data = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_directory, class_name, '*')
            images = glob(class_dir)
            lab = self.label_encoder.transform([class_name])[0]

            data = [(img, lab) for img in images]
            shuffle(data)

            train_part, test_part = train_test_split(data, train_size=train_size,
                                                     random_state=random_state)
            train_data.extend(train_part)
            test_data.extend(test_part)

        shuffle(train_data)
        shuffle(test_data)

        return np.array(train_data), np.array(test_data)


class BirdsDataGenerator(object):
    """
    Custom data generator for neural network.

    """
    def __init__(self, data):
        self.data = data
        self.batch_size = 30
        self.width = 256
        self.height = 256
        self.crop_width = 224
        self.crop_height = 224
        self.channels = 3

    def get_exploration_order(self):
        return np.random.permutation(range(len(self.data)))

    def data_generation(self, indexes_batch):
        x_data = np.empty((self.batch_size, self.crop_width, self.crop_height, self.channels))
        y_data = np.empty(self.batch_size, dtype=int)

        idx = 0
        for index in indexes_batch:
            image = Image.open(self.data[index][0]).resize((self.width, self.height), Image.LANCZOS)
            image = image.convert("RGB")
            w = np.random.randint(0, self.width - self.crop_width + 1)
            h = np.random.randint(0, self.height - self.crop_height + 1)
            image_crop = image.crop((w, h, w + self.crop_width, h + self.crop_height))

            x_data[idx, :, :, :] = np.array(image_crop)
            y_data[idx] = self.data[index][1]

            idx += 1

        return x_data, to_categorical(y_data, num_classes=50)

    def generate(self):
        data_no = self.data.shape[0]
        max_iter = int(data_no / self.batch_size)

        while True:
            indexes = self.get_exploration_order()
            for i in range(max_iter):
                indexes_batch = indexes[i * self.batch_size: (i + 1) * self.batch_size]
                x_data, y_data = self.data_generation(indexes_batch)
                yield x_data, y_data


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
