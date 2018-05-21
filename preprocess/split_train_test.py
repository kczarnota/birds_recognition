import os
from PIL import Image, ImageDraw
import argparse
import shutil
from random import randint
import random
from shutil import move

parser = argparse.ArgumentParser(description='SNR bounding box extraction and data argumentation.')
parser.add_argument("-ratio", default=0.8, type=float, help="Train/test size ratio")
parser.add_argument("-i", help="Input dir")
parser.add_argument("-o", help="Output dir")
args = parser.parse_args()

try:
    os.makedirs(os.path.join(args.o, 'train'))
    os.makedirs(os.path.join(args.o, 'test'))
except Exception as e:
    pass

for root, subdirs, files in os.walk(args.i):
    for d in subdirs:
        cls = d
        num_files_in_dir = 0
        list_of_files = []
        for root, subdirs, files in os.walk(os.path.join(args.i, d)):
            num_files_in_dir = len(files)
            list_of_files = files


        new_class_dir_train = os.path.join(args.o, 'train', cls)
        new_class_dir_test = os.path.join(args.o, 'test', cls)
        if not os.path.exists(new_class_dir_train):
            os.makedirs(new_class_dir_train)
        if not os.path.exists(new_class_dir_test):
            os.makedirs(new_class_dir_test)

        num_files_in_train = int(num_files_in_dir * args.ratio)
        num_files_in_test = num_files_in_dir - num_files_in_train

        random.shuffle(list_of_files)

        train_files = list_of_files[0:num_files_in_train]
        test_files = list_of_files[num_files_in_train:]

        for fn in train_files:
            move(os.path.join(args.i,  cls, fn), os.path.join(args.o, 'train', cls, fn))

        for fn in test_files:
            move(os.path.join(args.i,  cls, fn), os.path.join(args.o, 'test', cls, fn))






