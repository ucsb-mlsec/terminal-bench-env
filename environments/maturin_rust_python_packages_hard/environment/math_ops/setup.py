#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="math_ops",
    version="1.2.3",
    author="Math Team",
    author_email="math-team@example.com",
    description="High-performance mathematical operations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mathteam/math_ops",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Rust",
    ],
    rust_extensions=[
        RustExtension(
            "math_ops._rust_ops",
            binding=Binding.PyO3,
            debug=False,
        )
    ],
    install_requires=[],
    python_requires=">=3.8",
    zip_safe=False,
    include_package_data=True,
)
