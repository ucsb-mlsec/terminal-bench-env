FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/sketch2render/paired_samples

RUN pip3 install --no-cache-dir --break-system-packages torch torchvision pillow numpy

COPY sketch2render/architecture_requirements.txt /workspace/sketch2render/architecture_requirements.txt
COPY sketch2render/paired_samples/sketch_001.png /workspace/sketch2render/paired_samples/sketch_001.png
COPY sketch2render/paired_samples/render_001.png /workspace/sketch2render/paired_samples/render_001.png
COPY sketch2render/paired_samples/sketch_002.png /workspace/sketch2render/paired_samples/sketch_002.png
COPY sketch2render/paired_samples/render_002.png /workspace/sketch2render/paired_samples/render_002.png
COPY sketch2render/paired_samples/sketch_003.png /workspace/sketch2render/paired_samples/sketch_003.png
COPY sketch2render/paired_samples/render_003.png /workspace/sketch2render/paired_samples/render_003.png
COPY sketch2render/validation_input.png /workspace/sketch2render/validation_input.png

WORKDIR /workspace/sketch2render

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
