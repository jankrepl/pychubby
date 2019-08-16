[![Build Status](https://travis-ci.com/jankrepl/pychubby.svg?branch=master)](https://travis-ci.com/jankrepl/pychubby)
[![codecov](https://codecov.io/gh/jankrepl/pychubby/branch/master/graph/badge.svg)](https://codecov.io/gh/jankrepl/pychubby)

# PyChubby
![intro_resized](https://user-images.githubusercontent.com/18519371/63134578-59a81f00-bfca-11e9-9b75-45710f81c7f8.gif)

**Tool for creating chubby faces and way more!**

### Installation
##### 1) Install `pychubby` package
```bash
pip install git+https://github.com/jankrepl/pychubby.git
```
##### 2) Get a pretrained landmark detection model
```bash
git clone https://github.com/davisking/dlib-models.git  # ~400MB, takes some time to download
mkdir ~/.pychubby  # default folder where pychubby looks for trained models
cp dlib-models/shape_predictor_68_face_landmarks.dat.bz2 ~/.pychubby/
bzip2 -d ~/.pychubby/shape_predictor_68_face_landmarks.dat.bz2
```


### Description
For each face in an image define what **actions** are to be performed on it, `pychubby` will do the rest.

### Minimal Example
```python
import matplotlib.pyplot as plt

from pychubby.actions import Chubbify, Multiple, Pipeline, Smile
from pychubby.detect import LandmarkFace

img_path = 'path/to/your/image'
img = plt.imread(img_path)

lf = LandmarkFace.estimate(img)

a_per_face = Pipeline([Chubbify(), Smile()])
a_all = Multiple(a_per_face)


new_lf, _ = a_all.perform(lf)
new_lf.plot(show_landmarks=False, show_numbers=False)
```

### Development
```bash
git clone https://github.com/jankrepl/pychubby.git
cd pychubby
pip install -e .[dev]
```

