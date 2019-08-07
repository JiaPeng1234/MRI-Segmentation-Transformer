#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The python setup script.

"""

from setuptools import setup, find_packages


import os

setup(
    name='medseg_dl',
    description='Medical Image Segmentation',
    long_description=open(os.path.join(os.path.dirname(__file__), 'README.md')).read(),
    package_dir={'medseg_dl': 'medseg_dl'},
    packages=['medseg_dl', 'medseg_dl.model'],
    license='private',
    keywords='None',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Natural Language :: English',
        'Programming Language :: Python :: 3 :: Only'
    ],
    install_requires=[
        'numpy',
        'pydicom',
        'dicom2nifti',
        'nibabel',
        'matplotlib',
        'pyyaml'
    ],
)
