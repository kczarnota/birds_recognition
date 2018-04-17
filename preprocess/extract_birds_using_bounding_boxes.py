import os
from PIL import Image

image_size = 224


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
#
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
        try:
            cls = os.path.split(path)[1]
            new_class_dir = os.path.join(output_dir, cls)
            if not os.path.exists(new_class_dir):
                os.makedirs(new_class_dir)

            output_file_name = os.path.join(new_class_dir, f + ".jpg")

            orginal_image_file_name = os.path.join(path, f + ".jpg")
            img = Image.open(orginal_image_file_name)
            img2 = img.crop(bb_list[f])
            #img2 = img2.resize((image_size, image_size), Image.ANTIALIAS)
            rgb_image = Image.new("RGB", img2.size)
            rgb_image.paste(img2)
            rgb_image.save(output_file_name)
        except Exception as e:
            print(e)


fn = collect_files_names('../data/')
bb = load_bounding_boxes('../data/bounding_boxes.txt')

extract_bounding_boxes_from_images(bb, fn, '../data/BBPP')
