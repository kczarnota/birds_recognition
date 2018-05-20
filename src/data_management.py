import os
from glob import glob
from random import shuffle

import numpy as np
from PIL import Image
from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


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
    def __init__(self, data, batch_size=30):
        self.data = data
        self.batch_size = batch_size
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
