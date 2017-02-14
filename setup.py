#! /usr/bin/env python
"""
Author: Dario ML, bdevans
Program: SETUP.PY
Date: Saturday, June 06, 2014
Description: Setup and install TD algorithms.
"""

from distutils.core import setup

setup(
    name='python-aer',
    version='0.1.3',
    author="Dario Magliocchetti",
    author_email="darioml1911@gmail.com",
    url="https://github.com/bio-modelling/py-aer",
    description='Python Address Event Representation (AER) Library',
    long_description='''This package provides tools required to visualise,
                        manipulate and use address event representational data
                        (.aedat format). ''',
    packages=["pyaer"],
    license="GPL 2.0",
    classifiers=[
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Environment :: Console",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha"
    ],
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'Pillow'  # N.B. This cannot coexist with PIL
    ]
)
