#!/usr/bin/env python

from setuptools import setup, Extension

# Define the C extension module
mathops_extension = Extension(
    'mathops',
    sources=['mathops_ext.c']
)

setup(
    name='mathops',
    version='0.1.0',
    description='Optimized mathematical operations C extension module',
    author='Package Developer',
    ext_modules=[mathops_extension]
)