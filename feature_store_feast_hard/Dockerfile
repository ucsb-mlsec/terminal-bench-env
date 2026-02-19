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
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages feast pyarrow numpy

RUN pip3 install --no-cache-dir --break-system-packages \
    feast \
    pandas \
    pyarrow \
    fastparquet \
    numpy

RUN mkdir -p /workspace/data /workspace/feature_repo /tmp

COPY data/user_profiles.parquet /workspace/data/user_profiles.parquet
COPY data/transactions.csv /workspace/data/transactions.csv

WORKDIR /workspace