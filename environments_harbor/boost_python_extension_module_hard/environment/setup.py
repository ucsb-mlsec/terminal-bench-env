# DO NOT MODIFY - This file is correctly configured

from setuptools import setup, Extension
import sys

# Configuration for the C++ extension module
ext_modules = [
    Extension(
        'matrix_ops',  # Module name - must match BOOST_PYTHON_MODULE
        sources=['matrix_extension.cpp'],
        include_dirs=[
            '/usr/include/python3.10',
            '/usr/include',
        ],
        library_dirs=[
            '/usr/lib',
            '/usr/lib/x86_64-linux-gnu',
        ],
        libraries=['boost_python310'],
        extra_compile_args=['-std=c++11', '-fPIC'],
        language='c++'
    )
]

setup(
    name='matrix_ops',
    version='1.0',
    description='Matrix operations C++ extension module',
    author='Legacy Developer',
    ext_modules=ext_modules,
)