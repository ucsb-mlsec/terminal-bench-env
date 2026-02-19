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

RUN mkdir -p /workspace /workspace/data

COPY Snakefile /workspace/Snakefile
COPY data/sample1.fastq /workspace/data/sample1.fastq
COPY data/sample2.fastq /workspace/data/sample2.fastq
COPY data/sample3.fastq /workspace/data/sample3.fastq

WORKDIR /workspace