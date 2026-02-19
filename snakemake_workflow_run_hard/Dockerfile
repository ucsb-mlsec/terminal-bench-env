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

RUN pip3 install --no-cache-dir --break-system-packages snakemake

RUN mkdir -p /workspace/project/data/raw \
    /workspace/project/scripts \
    /workspace/project/config \
    /workspace/project/results/filtered \
    /workspace/project/results/analysis

COPY data/raw/sample1.fasta /workspace/project/data/raw/sample1.fasta
COPY data/raw/sample2.fasta /workspace/project/data/raw/sample2.fasta
COPY data/raw/sample3.fasta /workspace/project/data/raw/sample3.fasta

WORKDIR /workspace