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
    gdal-bin \
    libgdal-dev \
    python3-gdal && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gdal-bin libgdal-dev && \
    pip3 install --no-cache-dir --break-system-packages numpy rasterio && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages \
    numpy \
    rasterio

RUN mkdir -p /workspace/satellite_data /workspace/output

COPY data/satellite_data/red_band.txt /workspace/satellite_data/red_band.txt
COPY data/satellite_data/nir_band.txt /workspace/satellite_data/nir_band.txt
COPY data/satellite_data/quality_mask.txt /workspace/satellite_data/quality_mask.txt

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
