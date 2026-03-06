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
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages numpy

RUN mkdir -p /opt/cargo_system

COPY data/region_north.dat /opt/cargo_system/region_north.dat
COPY data/region_south.dat /opt/cargo_system/region_south.dat
COPY data/region_east.dat /opt/cargo_system/region_east.dat
COPY data/region_west.dat /opt/cargo_system/region_west.dat
COPY data/region_central.dat /opt/cargo_system/region_central.dat
COPY data/truck_specs.txt /opt/cargo_system/truck_specs.txt

WORKDIR /opt/cargo_system

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
