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


