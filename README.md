[![Build Status](https://travis-ci.com/jankrepl/pychubby.svg?branch=master)](https://travis-ci.com/jankrepl/pychubby)
[![codecov](https://codecov.io/gh/jankrepl/pychubby/branch/master/graph/badge.svg)](https://codecov.io/gh/jankrepl/pychubby)

# PyChubby
![](https://imgur.com/Qydlys5)

**Tool for creating chubby faces and way more!**

### Installation
##### 1) Install `pychubby` package
```bash
git clone https://github.com/jankrepl/pychubby.git
cd pychubby
pip install .
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

from pychubby.actions import Chubbify, Pipeline, Smile
from pychubby.detect import LandmarkFace

img_path = 'path/to/your/image'
img = plt.imread(img_path)

lf = LandmarkFace.estimate(img)
a = Pipeline([Chubbify(), Smile()])

new_lf, _ = a.perform(lf)
new_lf.plot()
```

### Development
```bash
git clone https://github.com/jankrepl/pychubby.git
cd pychubby
pip install -e .[dev]
```

