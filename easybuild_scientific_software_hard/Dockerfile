FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/easyconfigs /tmp

COPY easyconfigs/FFTW-3.3.8.eb /workspace/easyconfigs/FFTW-3.3.8.eb
COPY easyconfigs/HDF5-1.10.5.eb /workspace/easyconfigs/HDF5-1.10.5.eb
COPY easyconfigs/Python-3.8.2.eb /workspace/easyconfigs/Python-3.8.2.eb
COPY easyconfigs/NumPy-1.18.1-syntax-error.eb /workspace/easyconfigs/NumPy-1.18.1-syntax-error.eb
COPY easyconfigs/SciPy-1.4.1-missing-fields.eb /workspace/easyconfigs/SciPy-1.4.1-missing-fields.eb
COPY easyconfigs/OpenMPI-4.0.3-invalid-toolchain.eb /workspace/easyconfigs/OpenMPI-4.0.3-invalid-toolchain.eb
COPY easyconfigs/CMake-3.16.4-wrong-types.eb /workspace/easyconfigs/CMake-3.16.4-wrong-types.eb
COPY easyconfigs/Boost-1.72.0.eb /workspace/easyconfigs/Boost-1.72.0.eb
COPY easyconfigs/GROMACS-2020.1-unclosed-bracket.eb /workspace/easyconfigs/GROMACS-2020.1-unclosed-bracket.eb
COPY easyconfigs/TensorFlow-2.3.1-no-description.eb /workspace/easyconfigs/TensorFlow-2.3.1-no-description.eb
COPY easyconfigs/GCC-9.3.0.eb /workspace/easyconfigs/GCC-9.3.0.eb
COPY easyconfigs/NetCDF-4.7.4-malformed-deps.eb /workspace/easyconfigs/NetCDF-4.7.4-malformed-deps.eb

WORKDIR /workspace