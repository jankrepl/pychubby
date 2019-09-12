from setuptools import setup
import os

import pychubby

INSTALL_REQUIRES = [
    "click>=7.0",
    "matplotlib>=2.0.0",
    "numpy>=1.16.4",
    "opencv-python>=4.1.0.25",
    "scikit-image",
]

if "RTD_BUILD" not in os.environ:
    # ReadTheDocs cannot handle compilation
    INSTALL_REQUIRES += ["dlib"]

LONG_DESCRIPTION = "Automated face warping tool"
PROJECT_URLS = {
    "Bug Tracker": "https://github.com/jankrepl/pychubby/issues",
    "Documentation": "https://pychubby.readthedocs.io",
    "Source Code": "https://github.com/jankrepl/pychubby",
}
VERSION = pychubby.__version__

setup(
    name="pychubby",
    version=VERSION,
    author="Jan Krepl",
    author_email="kjan.official@gmail.com",
    description="Automated face warping tool",
    long_description=LONG_DESCRIPTION,
    url="https://github.com/jankrepl/pychubby",
    project_urls=PROJECT_URLS,
    packages=["pychubby"],
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: C",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        ("Programming Language :: Python :: " "Implementation :: CPython"),
    ],
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "dev": ["codecov", "flake8", "pydocstyle", "pytest>=3.6", "pytest-cov", "tox"],
        "docs": ["sphinx", "sphinx_rtd_theme"],
    },
    entry_points={"console_scripts": ["pc = pychubby.cli:cli"]},
)
