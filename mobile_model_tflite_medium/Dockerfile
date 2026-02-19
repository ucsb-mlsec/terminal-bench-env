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
RUN pip3 install --no-cache-dir --break-system-packages tensorflow>=2.0 numpy

RUN pip3 install --no-cache-dir --break-system-packages \
    tensorflow \
    numpy

RUN mkdir -p /workspace/models/classifier /workspace/test_data /workspace/output

COPY data/saved_model.pb /workspace/models/classifier/saved_model.pb
COPY data/variables/variables.index /workspace/models/classifier/variables/variables.index
COPY data/variables/variables.data-00000-of-00001 /workspace/models/classifier/variables/variables.data-00000-of-00001
COPY data/sample_image.npy /workspace/test_data/sample_image.npy

WORKDIR /workspace