[![Build Status](https://travis-ci.com/jankrepl/pychubby.svg?branch=master)](https://travis-ci.com/jankrepl/pychubby)
[![codecov](https://codecov.io/gh/jankrepl/pychubby/branch/master/graph/badge.svg)](https://codecov.io/gh/jankrepl/pychubby)
[![PyPI version](https://badge.fury.io/py/pychubby.svg)](https://badge.fury.io/py/pychubby)
[![Documentation Status](https://readthedocs.org/projects/pychubby/badge/?version=latest)](https://pychubby.readthedocs.io/en/latest/?badge=latest)

# PyChubby
**Tool for automated face warping**

![intro](https://user-images.githubusercontent.com/18519371/64875224-ed621f00-d64c-11e9-92ad-8f76a4b95bcc.gif)

### Installation
```bash
pip install pychubby
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

### CLI
`pychubby` also comes with a CLI that exposes some
of its functionality. You can list the commands with `pc --help`:

```text
Usage: pc [OPTIONS] COMMAND [ARGS]...

  Automated face warping tool.

Options:
  --help  Show this message and exit.

Commands:
  list     List available actions.
  perform  Take an action.
```

To perform an action (Smile in the example below) and plot the result on the screen 
```bash
pc perform Smile INPUT_IMG_PATH
```

or if you want to create a new image and save it
```bash
pc perform Smile INPUT_IMG_PATH OUTPUT_IMG_PATH
```

### Development
```bash
git clone https://github.com/jankrepl/pychubby.git
cd pychubby
pip install -e .[dev]
```

