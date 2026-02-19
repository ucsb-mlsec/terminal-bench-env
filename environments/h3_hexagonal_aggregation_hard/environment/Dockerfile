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
    apt-get install -y python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages pandas h3

RUN mkdir -p /data/rides /solution

COPY data/rides/san_francisco.csv /data/rides/san_francisco.csv
COPY data/rides/new_york.csv /data/rides/new_york.csv
COPY data/rides/chicago.csv /data/rides/chicago.csv

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
