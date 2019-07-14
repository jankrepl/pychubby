from setuptools import setup

setup(name='pychubby',
      version=0.1,
      packages=['pychubby'],
      install_requires=[
                        'click>=7.0',
                        'matplotlib>=2.0.0',
                        'numpy>=1.16.4',
                        'opencv-python>=4.1.0.25',
                        'scikit-image' 
                        ],
      extras_require={'dev': [
                              'codecov',
                              'flake8',
                              'pytest>=3.6',
                              'pytest-cov',
                              'tox'
                              ]
                              },
      entry_points={'console_scripts': ['pc = pychubby.cli:cli']}
      )
