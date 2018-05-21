import os
from PIL import Image, ImageDraw
import argparse
import shutil
from random import randint
import random


parser = argparse.ArgumentParser(description='SNR bounding box extraction and data argumentation.')
parser.add_argument("-i", help="Input dir")
parser.add_argument("-b", help="Bounding boxes file")
parser.add_argument("-o", help="Output dir")
parser.add_argument("-f", default=False, required=False, action='store_true', help="Force overwrite")
parser.add_argument("--translate", default=0, required=False, type=int, help="How many times to do translate argumentation")
parser.add_argument("--rotate", default=0, required=False, type=int, help="How many times to do rotate argumentation")
parser.add_argument("--translate-rotate", default=0, required=False, type=int, help="How many times to do translate + rotate argumentation")
parser.add_argument("--debug", default=False, required=False, action='store_true', help="Debug mode")
parser.add_argument("--square", default=False, required=False, action='store_true', help="Crop square")
args = parser.parse_args()

image_size = 224

def draw_rectangle(draw, coordinates, color, width=1):
    for i in range(width):
        rect_start = (coordinates[0][0] - i, coordinates[0][1] - i)
        rect_end = (coordinates[1][0] + i, coordinates[1][1] + i)
        draw.rectangle((rect_start, rect_end), outline = color)


def fit_bounding_box(image_size, orginal_bounding_box):
    x, y, w, h = orginal_bounding_box

    width = w - x
    height = h - y

    max_wh = max([width, height])

    middle_x = x + width//2
    middle_y = y + height//2

    if args.square:
        if middle_x - max_wh//2 < 0:
            max_wh -= (max_wh//2 - middle_x)*2

        if middle_x + max_wh//2 > image_size[0]:
            max_wh -= ((max_wh//2 + middle_x) - image_size[0])*2 + 1

        if middle_y - max_wh//2 < 0:
            max_wh -= (max_wh//2 - middle_y)*2 + 1

        if middle_y + max_wh//2 > image_size[1]:
            max_wh -= ((max_wh//2 + middle_y) - image_size[1])*2 + 1

    new_x = middle_x - max_wh//2
    new_y = middle_y - max_wh//2

    new_w = new_x + max_wh
    new_h = new_y + max_wh

    return (new_x, new_y, new_w, new_h)


def load_bounding_boxes(file_name):
    bb = {}  # map form filename to bouding box
    with open(file_name) as f:
        for line in f:
            line = line.strip()
            filename, x, y, w, h = line.split(" ")
            x = int(x)
            y = int(y)
            width = int(w)
            height = int(h)
            w = width + x
            h = height + y
            bb[filename.replace('-', '')] = (x, y, w, h)
    return bb


def collect_files_names(root_dir):
    coll = {}
    for root, subdirs, files in os.walk(root_dir):
        for f in files:
            f, ext = os.path.splitext(f)  # remove jpg extension
            if ext == '.jpg':
                coll[f] = root

    return coll

def do_argumentation(img, box):
    images, bb = [], []
    images.append(img)
    bb.append(box)

    for i in range(args.translate):
        vec = (randint(1, 30), randint(1, 30))
        #import ipdb; ipdb.set_trace()
        new_bb = (box[0] + vec[0], box[1] + vec[1], box[2] + vec[0], box[3] + vec[1])
        images.append(img)
        bb.append(new_bb)

    for i in range(args.rotate):
        angle = random.uniform(0.1, 5)
        ni = img.copy()
        ni = ni.rotate(angle)
        images.append(ni)
        bb.append(box)

    for i in range(args.translate_rotate):
        angle = random.uniform(0.1, 5)
        ni = img.rotate(angle)
        vec = (randint(1, 30), randint(1, 30))
        new_bb = (box[0] + vec[0], box[1] + vec[1], box[2] + vec[0], box[3] + vec[1])
        images.append(ni)
        bb.append(new_bb)

    return images, bb

def extract_bounding_boxes_from_images(bb_list, file_names, output_dir):
    for f, path in file_names.items():
        print(f)
        cls = os.path.split(path)[1]
        new_class_dir = os.path.join(output_dir, cls)
        if not os.path.exists(new_class_dir):
            os.makedirs(new_class_dir)


        orginal_image_file_name = os.path.join(path, f + ".jpg")
        img = Image.open(orginal_image_file_name)
        argumented_images, argumented_bounding_boxes = do_argumentation(img, bb_list[f])
        i = 0
        for image, box in zip(argumented_images, argumented_bounding_boxes):
            print(i)
            new_bb = fit_bounding_box(image.size, box)
            output_file_name = os.path.join(new_class_dir, f + str(i) + ".jpg")
            i += 1

            if not args.debug:
                img2 = image.crop(new_bb)
                img2 = img2.resize((image_size, image_size), Image.ANTIALIAS)
                rgb_image = Image.new("RGB", img2.size)
                rgb_image.paste(img2)
                rgb_image.save(output_file_name)
            else:
                rgb_image = Image.new("RGB", img.size)
                rgb_image.paste(image)
                draw = ImageDraw.Draw(rgb_image)
                draw_rectangle(draw, [(new_bb[0], new_bb[1]), (new_bb[2], new_bb[3])], color=(255,0,0), width=5)
                del draw
                rgb_image.save(output_file_name)




fn = collect_files_names(args.i)
bb = load_bounding_boxes(args.b)

if os.path.exists(args.o):
    if not args.f:
        raise Exception("Output dir already exists")
else:
    os.makedirs(args.o)

extract_bounding_boxes_from_images(bb, fn, args.o)
