[![Build Status](https://travis-ci.com/jankrepl/pychubby.svg?branch=master)](https://travis-ci.com/jankrepl/pychubby)
[![codecov](https://codecov.io/gh/jankrepl/pychubby/branch/master/graph/badge.svg)](https://codecov.io/gh/jankrepl/pychubby)

# PyChubby
**Tool for automated face warping**

![intro_resized](https://user-images.githubusercontent.com/18519371/63134578-59a81f00-bfca-11e9-9b75-45710f81c7f8.gif)


### Installation
```bash
pip install git+https://github.com/jankrepl/pychubby.git
```
### Description
For each face in an image define what **actions** are to be performed on it, `pychubby` will do the rest.

### Documentation
<https://pychubby.readthedocs.io>

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

