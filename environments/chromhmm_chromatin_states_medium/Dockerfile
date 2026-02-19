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
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages numpy

RUN pip3 install --no-cache-dir --break-system-packages pandas

RUN mkdir -p /data/chipseq/CellA /data/chipseq/CellB /solution

COPY data/chipseq/CellA/H3K4me3.bed /data/chipseq/CellA/H3K4me3.bed
COPY data/chipseq/CellA/H3K27ac.bed /data/chipseq/CellA/H3K27ac.bed
COPY data/chipseq/CellA/H3K36me3.bed /data/chipseq/CellA/H3K36me3.bed
COPY data/chipseq/CellB/H3K4me3.bed /data/chipseq/CellB/H3K4me3.bed
COPY data/chipseq/CellB/H3K27ac.bed /data/chipseq/CellB/H3K27ac.bed
COPY data/chipseq/CellB/H3K36me3.bed /data/chipseq/CellB/H3K36me3.bed

WORKDIR /workspace