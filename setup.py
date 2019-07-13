from setuptools import setup

setup(name='pychubby',
      version=0.1,
      packages=['pychubby'],
      install_requires=['click'],
      extras_require={'dev': ['flake8', 'pytest', 'pytest-cov', 'tox']},
      entry_points={'console_scripts': ['pc = pychubby.cli:cli'] }
      )
