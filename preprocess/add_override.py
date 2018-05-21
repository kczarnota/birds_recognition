import os
from PIL import Image, ImageDraw
import argparse
import shutil
from random import randint
import random

parser = argparse.ArgumentParser(description='Add random black rectangle for every file in input dir.')
parser.add_argument("-i", help="Input dir")
args = parser.parse_args()

image_size = 224

for root, subdirs, files in os.walk(args.i):
    for f in files:
        image_name = os.path.join(root, f)
        img = Image.open(image_name)
        draw = ImageDraw.Draw(img)

        image_width, image_height = img.size

        rect =  (randint(0, image_width), randint(0, image_height), randint(1, image_width), randint(1, image_height))
        draw.rectangle(rect, fill=(0,0,0))

        del draw
        img.save(image_name)

