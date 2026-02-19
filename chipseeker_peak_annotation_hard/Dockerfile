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
    gzip \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y bedtools && \
    pip3 install --no-cache-dir --break-system-packages pybedtools pysam numpy && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages \
    pandas \
    numpy \
    pybedtools \
    pyranges

RUN mkdir -p /data/chipseq_peaks /data/reference /results

COPY data/chipseq_peaks/control.bed /data/chipseq_peaks/control.bed
COPY data/chipseq_peaks/treatment_24h.bed /data/chipseq_peaks/treatment_24h.bed
COPY data/chipseq_peaks/treatment_48h.bed /data/chipseq_peaks/treatment_48h.bed
COPY data/reference/genome_annotation.gtf.gz /data/reference/genome_annotation.gtf.gz

WORKDIR /workspace