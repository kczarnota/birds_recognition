import os
from PIL import Image, ImageDraw
import argparse
import shutil


parser = argparse.ArgumentParser(description='SNR bounding box extraction and data argumentation.')
parser.add_argument("-i", help="Input dir")
parser.add_argument("-b", help="Bounding boxes file")
parser.add_argument("-o", help="Output dir")
parser.add_argument("-f", default=False, required=False, action='store_true', help="Force overwrite")
parser.add_argument("--debug", default=False, required=False, action='store_true', help="Debug mode")

args = parser.parse_args()

image_size = 224

def draw_rectangle(draw, coordinates, color, width=1):
    for i in range(width):
        rect_start = (coordinates[0][0] - i, coordinates[0][1] - i)
        rect_end = (coordinates[1][0] + i, coordinates[1][1] + i)
        draw.rectangle((rect_start, rect_end), outline = color)


def fit_bounding_box(image_size, orginal_bounding_box):
#            if width != height:
#                if width > height:
#                    bigger = width
#                    diff = (width - height)//2
#                    y -= diff
#                    h += diff
#                    new_height = h - y
#                    if new_height != width:
#                        h += abs(new_height - width)
#                else:
#                    bigger = height
#                    diff = (height- width)//2
#                    x -= diff
#                    w += diff
#                    new_width = w - x
#                    if new_width != height:
#                        w += abs(new_width - height)
    return orginal_bounding_box


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


def extract_bounding_boxes_from_images(bb_list, file_names, output_dir):
    for f, path in file_names.items():
        print(f)
        try:
            cls = os.path.split(path)[1]
            new_class_dir = os.path.join(output_dir, cls)
            if not os.path.exists(new_class_dir):
                os.makedirs(new_class_dir)

            output_file_name = os.path.join(new_class_dir, f + ".jpg")

            orginal_image_file_name = os.path.join(path, f + ".jpg")
            img = Image.open(orginal_image_file_name)
            new_bb = fit_bounding_box(img.size, bb_list[f])
            if not args.debug:
                img2 = img.crop(new_bb)
                img2 = img2.resize((image_size, image_size), Image.ANTIALIAS)
                rgb_image = Image.new("RGB", img2.size)
                rgb_image.paste(img2)
                rgb_image.save(output_file_name)
            else:
                rgb_image = Image.new("RGB", img.size)
                rgb_image.paste(img)
                draw = ImageDraw.Draw(rgb_image)
                draw_rectangle(draw, [(new_bb[0], new_bb[1]), (new_bb[2], new_bb[3])], color=(255,0,0), width=5)
                del draw
                rgb_image.save(output_file_name)


        except Exception as e:
            print(e)


fn = collect_files_names(args.i)
bb = load_bounding_boxes(args.b)

if os.path.exists(args.o):
    if not args.f:
        raise Exception("Output dir already exists")
    else:
        for root, dirs, files in os.walk(args.o):
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))

else:
    os.makedirs(args.o)

extract_bounding_boxes_from_images(bb, fn, args.o)