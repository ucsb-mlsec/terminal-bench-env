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
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages pillow numpy

RUN pip3 install --no-cache-dir --break-system-packages \
    numpy \
    pillow

RUN mkdir -p /workspace/normal_maps /workspace/reference

COPY data/normal_maps/candidate_001.png /workspace/normal_maps/candidate_001.png
COPY data/normal_maps/candidate_002.png /workspace/normal_maps/candidate_002.png
COPY data/normal_maps/candidate_003.png /workspace/normal_maps/candidate_003.png
COPY data/normal_maps/candidate_004.png /workspace/normal_maps/candidate_004.png
COPY data/normal_maps/candidate_005.png /workspace/normal_maps/candidate_005.png
COPY data/normal_maps/candidate_006.png /workspace/normal_maps/candidate_006.png
COPY data/normal_maps/candidate_007.png /workspace/normal_maps/candidate_007.png
COPY data/normal_maps/candidate_008.png /workspace/normal_maps/candidate_008.png
COPY data/normal_maps/candidate_009.png /workspace/normal_maps/candidate_009.png
COPY data/normal_maps/candidate_010.png /workspace/normal_maps/candidate_010.png
COPY data/reference/normal_map_spec.txt /workspace/reference/normal_map_spec.txt

WORKDIR /workspace