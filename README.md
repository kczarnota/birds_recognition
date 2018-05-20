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

## Convnet 
```bash
python train_convnet.py
```