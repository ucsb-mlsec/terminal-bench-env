FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/cache/binaries /solution

COPY data/pipeline_manifest.txt /workspace/cache/pipeline_manifest.txt
COPY data/binaries/vert_001.cache /workspace/cache/binaries/vert_001.cache
COPY data/binaries/frag_042.cache /workspace/cache/binaries/frag_042.cache
COPY data/binaries/comp_015.cache /workspace/cache/binaries/comp_015.cache
COPY data/binaries/geom_008.cache /workspace/cache/binaries/geom_008.cache
COPY data/binaries/tess_003.cache /workspace/cache/binaries/tess_003.cache
COPY data/binaries/vert_020.cache /workspace/cache/binaries/vert_020.cache
COPY data/binaries/frag_051.cache /workspace/cache/binaries/frag_051.cache
COPY data/binaries/comp_022.cache /workspace/cache/binaries/comp_022.cache
COPY data/binaries/vert_035.cache /workspace/cache/binaries/vert_035.cache
COPY data/binaries/frag_067.cache /workspace/cache/binaries/frag_067.cache
COPY data/binaries/geom_012.cache /workspace/cache/binaries/geom_012.cache
COPY data/binaries/comp_040.cache /workspace/cache/binaries/comp_040.cache
COPY data/binaries/vert_055.cache /workspace/cache/binaries/vert_055.cache
COPY data/binaries/frag_088.cache /workspace/cache/binaries/frag_088.cache
COPY data/binaries/tess_009.cache /workspace/cache/binaries/tess_009.cache

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
