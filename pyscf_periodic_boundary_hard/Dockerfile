FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN apt-get update && \
    apt-get install -y \
    python3-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    cmake \
    git && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages pyscf numpy

RUN pip3 install --no-cache-dir --break-system-packages \
    numpy \
    scipy \
    h5py \
    pyscf

RUN mkdir -p /workspace

COPY data/silicon_data.txt /workspace/silicon_data.txt
COPY template.py /workspace/template.py

WORKDIR /workspace