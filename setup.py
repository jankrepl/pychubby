from setuptools import setup

setup(name='pychubby',
      version=0.1,
      packages=['pychubby'],
      install_requires=[],
      extras_require={'dev': ['flake8', 'pytest', 'tox']}
      )
