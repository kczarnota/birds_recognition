import os
from sklearn import svm
from numpy import genfromtxt
import numpy as np
import argparse
import matplotlib.pyplot as plt
import keras
from keras.models import Model
from keras.preprocessing.image import img_to_array
from PIL import Image

def normalize(v):
    norm=np.linalg.norm(v, ord=1)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

parser = argparse.ArgumentParser(description='Compute CNN features from last layer')
parser.add_argument("-m", required=True, help="Path to model")
parser.add_argument("-i", required=True, help="Dir containig images for feautres computation")

args = parser.parse_args()

model = keras.models.load_model(args.m)

#print(model.summary())

layer_name = 'dense_3'

intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)

for root, subdirs, files in os.walk(args.i):
    cls_id = 0
    for d in subdirs:
        cls = d
        num_files_in_dir = 0
        list_of_files = []
        for root, subdirs, files in os.walk(os.path.join(args.i, d)):
            num_files_in_dir = len(files)
            list_of_files = files

        for f in list_of_files:

            file_path = os.path.join(args.i, d, f)
            img = Image.open(file_path)
            if (int(model.input.shape[1]),int(model.input.shape[2])) != img.size:
                img = img.resize((int(model.input.shape[1]),int(model.input.shape[2])))
            rgb = Image.new("RGB", img.size, (255, 255, 255))
            rgb.paste(img)

            arr = img_to_array(rgb)
            intermediate_output = intermediate_layer_model.predict(np.array([arr]))
            intermediate_output = normalize(intermediate_output.flatten())
            intermediate_output = intermediate_output.tolist()
            csv_line = ",".join([str(i) for i in intermediate_output]) + "," + str(cls_id)
            print(csv_line)

        cls_id += 1
