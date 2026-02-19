from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "vector_ops",
        ["vector_ops.pyx"]
    )
]

setup(
    name="vector_ops",
    ext_modules=cythonize(extensions)
)