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
RUN apt-get update && apt-get install -y proj-bin libproj-dev && \
    pip3 install --no-cache-dir --break-system-packages pyproj>=3.0 shapely>=2.0 && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages \
    pyproj \
    pandas \
    numpy \
    shapely

RUN mkdir -p /workspace/geodata /workspace/output

COPY data/geodata/gps_points.csv /workspace/geodata/gps_points.csv
COPY data/geodata/survey_points.csv /workspace/geodata/survey_points.csv
COPY data/geodata/municipal_points.csv /workspace/geodata/municipal_points.csv

WORKDIR /workspace