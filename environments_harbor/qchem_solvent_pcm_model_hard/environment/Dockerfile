FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /data /tmp

COPY data/molecule1_water.in /data/molecule1_water.in
COPY data/molecule1_acetonitrile.in /data/molecule1_acetonitrile.in
COPY data/molecule2_water.in /data/molecule2_water.in
COPY data/molecule2_chloroform.in /data/molecule2_chloroform.in
COPY data/molecule3_hexane.in /data/molecule3_hexane.in
COPY data/molecule3_acetonitrile.in /data/molecule3_acetonitrile.in
COPY data/molecule4_water.in /data/molecule4_water.in
COPY data/molecule4_chloroform.in /data/molecule4_chloroform.in

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
