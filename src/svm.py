from sklearn import svm
from numpy import genfromtxt
import numpy as np
import argparse
import matplotlib.pyplot as plt

def load_data(csv_data_file_name, train_ratio=0.8):
    array = genfromtxt(csv_data_file_name, delimiter=',')
    class_labels = np.unique(array[:,-1].astype(np.int32))
    print("Data loading done")
    print("Loaded {} samples splitted into {} classes".format(array.shape[0], class_labels.shape[0]))
    print("Splitting each class by {ratio} ratio".format(ratio=train_ratio))

    x_train = np.empty((0, array.shape[1] - 1))
    y_train = np.empty((0,1))
    x_test = np.empty((0, array.shape[1] - 1))
    y_test = np.empty((0,1))

    for c in np.nditer(class_labels):
        row_filter = array[:,-1].astype(np.int32) == c
        class_samples = array[row_filter]
        np.random.shuffle(class_samples)

        training_set_size = int(train_ratio * class_samples.shape[0])

        if float(training_set_size) != train_ratio * class_samples.shape[0]:
            training_set_size += 1

        x_train_class = class_samples[:training_set_size, :-1]
        y_train_class = class_samples[:training_set_size, -1:].astype(dtype=np.int32)

        x_test_class = class_samples[training_set_size:, :-1]
        y_test_class = class_samples[training_set_size:, -1:].astype(dtype=np.int32)

        x_train = np.append(x_train, x_train_class, axis=0)
        y_train = np.append(y_train, y_train_class, axis=0)

        x_test = np.append(x_test, x_test_class, axis=0)
        y_test = np.append(y_test, y_test_class, axis=0)

    return x_train, y_train.flatten(), x_test, y_test.flatten()

parser = argparse.ArgumentParser(description='Train SVM classifier')
parser.add_argument("-i", required=True, help="Path to data")
parser.add_argument("-r", default=0.8, type=float, help="Train/Test split ratio")
parser.add_argument("-d", default=1, type=int, help="SVM Poly kernel degree")

args = parser.parse_args()

train_x, train_y, test_x, test_y = load_data(args.i, args.r)

#this is SVM one-vs-one classifier
clf = svm.SVC(kernel='poly', degree=args.d)
clf.fit(train_x, train_y)
print(clf.score(train_x, train_y))
print(clf.score(test_x, test_y))


