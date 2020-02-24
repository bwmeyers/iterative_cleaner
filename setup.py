#!/usr/bin/env python
from setuptools import setup
import sys

reqs = ['numpy>=1.16.0', 'matplotlib>=3.0.0', 'scipy>=1.2.0']

if sys.version_info < (3, 5):
    sys.exit('iterative_cleaner requires Python 3.5+')

setup(
    name='iterative_cleaner',
    author='Bradley Meyers',
    author_email='bradley.meyers1993@gmail.com',
    description='A package to clean RFI from folded pulsar data',
    keywords="signal processing",
    install_requires=reqs,
    packages=['iterative_cleaner'],
    scripts=['scripts/iterative_cleaner']
)
