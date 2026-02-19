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
    python3-pip \
    gdal-bin \
    libgdal-dev \
    libspatialindex-dev && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y libgeos-dev libproj-dev gdal-bin && \
    pip3 install --no-cache-dir --break-system-packages geopandas shapely pyproj && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages \
    geopandas \
    shapely \
    fiona \
    pyproj \
    rtree

RUN mkdir -p /data /output

COPY data/neighborhoods.geojson /data/neighborhoods.geojson
COPY data/development_zones.geojson /data/development_zones.geojson
COPY data/city_boundary.geojson /data/city_boundary.geojson

WORKDIR /workspace