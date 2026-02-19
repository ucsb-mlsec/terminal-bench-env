FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/renderer /workspace/test_data

RUN pip3 install --no-cache-dir --break-system-packages numpy

COPY renderer/volume_renderer.py /workspace/renderer/volume_renderer.py
COPY renderer/test_renderer.py /workspace/renderer/test_renderer.py
COPY data/sphere_8x8x8.vol /workspace/test_data/sphere_8x8x8.vol
COPY data/uniform_8x8x8.vol /workspace/test_data/uniform_8x8x8.vol
COPY data/gradient_16x16x16.vol /workspace/test_data/gradient_16x16x16.vol
COPY data/box_8x8x8.vol /workspace/test_data/box_8x8x8.vol

WORKDIR /workspace