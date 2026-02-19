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
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages numpy

RUN pip3 install --no-cache-dir --break-system-packages \
    numpy \
    pandas \
    scipy

RUN mkdir -p /data/AutoPick/job003 /data/micrographs /data /output

COPY data/AutoPick/job003/autopick.star /data/AutoPick/job003/autopick.star
COPY data/micrographs/quality_metrics.star /data/micrographs/quality_metrics.star
COPY data/reference_manual_picks.star /data/reference_manual_picks.star

WORKDIR /workspace