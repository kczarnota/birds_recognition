# SNR Project - recognizing birds
## Setup
We're providing requirements.txt file, which can be used to prepare environment using virtualenv on conda

**conda**
```bash
conda create --name birds python=3.6
source activate birds
conda install --yes --file requirements.txt
```
**virtualenv**
```bash
virtualenv --python=/path/to/python3.6 birds
source birds/bin/activate
pip install -r requirements.txt
```

You need to install octave package:
```bash
sudo apt-get octave liboctave-dev
octave-cli
pkg install -forge image
```
## Perceptron experiment
Put SET_A and bounding_boxes.txt in data directory

```bash
# from preprocess directory
python extract_birds_using_bounding_boxes.py

# from BSIF - compute BSIF features for images
octave-cli bsif_maker.m
python mat2csv.py bsifhistnorm_features_gray_cube.mat > bsifhistnorm_features_gray_cube.csv
octave-cli bsif_maker_rgb.m
python mat2csv.py bsifhistnorm_features_rgb_cube.mat > bsifhistnorm_features_rgb_cube.csv

# from src
python perceptron_experiment.py
```

## Preprocessing scripts

- add_overrite.py - draws random rectangle on every image from dir
    ```
    usage: add_override.py [-h] [-i I]

    Add random black rectangle for every file in input dir.

    optional arguments:
    -h, --help  show this help message and exit
    -i I        Input dir
    ```

- extract.py - script for crop extraction using bounding boxes and for static data argumentation
    ```
    usage: extract.py [-h] [-i I] [-b B] [-o O] [-f] [--translate TRANSLATE]
                    [--rotate ROTATE] [--translate-rotate TRANSLATE_ROTATE]
                    [--debug] [--square] [--noise]

    Bounding box extraction and data argumentation.

    optional arguments:
    -h, --help            show this help message and exit
    -i I                  Input dir
    -b B                  Bounding boxes file
    -o O                  Output dir
    -f                    Force overwrite
    --translate TRANSLATE
                            How many times to do translate argumentation
    --rotate ROTATE       How many times to do rotate argumentation
    --translate-rotate TRANSLATE_ROTATE
                            How many times to do translate + rotate argumentation
    --debug               Debug mode
    --square              Crop square
    --noise               Add noise

    ```
- spilt_train_test.py - moves files from input dir to output and splits them between train and test dirs using ratio argument
    ```
    usage: split_train_test.py [-h] [-ratio RATIO] [-i I] [-o O]

    Split data between test and train sets by ratio.

    optional arguments:
    -h, --help    show this help message and exit
    -ratio RATIO  Train/test size ratio
    -i I          Input dir
    -o O          Output dir

    ```

## Convnet
```bash
python train_convnet.py
```
