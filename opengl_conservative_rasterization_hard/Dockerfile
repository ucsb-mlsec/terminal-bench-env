FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/config /workspace/docs

COPY config/pipeline.conf /workspace/config/pipeline.conf
COPY config/extensions.conf /workspace/config/extensions.conf
COPY config/rasterizer.conf /workspace/config/rasterizer.conf
COPY docs/conservative_raster_spec.txt /workspace/docs/conservative_raster_spec.txt
COPY validator.py /workspace/validator.py

WORKDIR /workspace