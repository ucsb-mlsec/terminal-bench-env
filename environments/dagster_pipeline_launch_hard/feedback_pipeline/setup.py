from setuptools import setup, find_packages

setup(
    name='feedback_pipeline',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'dagster',
        'pandas',
    ],
)