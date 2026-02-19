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
    bedtools \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y bedtools && \
    pip3 install --no-cache-dir --break-system-packages pybedtools && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages pandas numpy

RUN mkdir -p /workspace/peaks /workspace/reference /output

COPY data/peaks/called_peaks.bed /workspace/peaks/called_peaks.bed
COPY data/peaks/fragment_counts.tsv /workspace/peaks/fragment_counts.tsv
COPY data/peaks/genome_sizes.txt /workspace/peaks/genome_sizes.txt
COPY data/reference/blacklist.bed /workspace/reference/blacklist.bed

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
