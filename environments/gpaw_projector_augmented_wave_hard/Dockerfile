FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/gpaw_outputs

COPY data/gpaw_outputs/h2o_grid_010.txt /workspace/gpaw_outputs/h2o_grid_010.txt
COPY data/gpaw_outputs/h2o_grid_012.txt /workspace/gpaw_outputs/h2o_grid_012.txt
COPY data/gpaw_outputs/h2o_grid_015.txt /workspace/gpaw_outputs/h2o_grid_015.txt
COPY data/gpaw_outputs/h2o_grid_018.txt /workspace/gpaw_outputs/h2o_grid_018.txt
COPY data/gpaw_outputs/h2o_grid_020.txt /workspace/gpaw_outputs/h2o_grid_020.txt
COPY data/gpaw_outputs/h2o_grid_025.txt /workspace/gpaw_outputs/h2o_grid_025.txt

WORKDIR /workspace