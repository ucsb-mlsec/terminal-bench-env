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
    apt-get install -y --no-install-recommends \
    python3-pip \
    libnetcdf-dev \
    libhdf5-dev && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y libnetcdf-dev libhdf5-dev && \
    pip3 install --no-cache-dir --break-system-packages netCDF4 numpy && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages \
    netCDF4 \
    numpy

RUN mkdir -p /data/climate /output

COPY data/climate/temp_2021.nc /data/climate/temp_2021.nc
COPY data/climate/temp_2022.nc /data/climate/temp_2022.nc
COPY data/climate/temp_2023.nc /data/climate/temp_2023.nc

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
