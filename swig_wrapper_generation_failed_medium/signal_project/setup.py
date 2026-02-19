#!/usr/bin/env python

from distutils.core import setup, Extension

signal_module = Extension(
    '_signal_module',
    sources=['signal.i', 'signal_ops.c'],
)

setup(
    name='signal_module',
    version='1.0',
    ext_modules=[signal_module],
)