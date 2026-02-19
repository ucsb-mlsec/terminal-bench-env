from setuptools import setup, Extension

# Define the C++ extension module
stringtools_module = Extension(
    'stringtoolz',  # BUG: Wrong module name (should be 'stringtools')
    sources=['stringtools.cxx'],  # BUG: Wrong file extension (should be .cpp)
    include_dirs=['./include'],  # BUG: Wrong include directory path
    language='c++',
    extra_compile_args=['-std=c++11'],
)

setup(
    name='stringtools',
    version='0.1.0',
    description='String processing tools using C++',
    ext_modules=[stringtools_module],
    python_requires='>=3.6',
)