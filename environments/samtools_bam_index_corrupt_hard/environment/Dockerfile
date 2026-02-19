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
    apt-get install -y samtools && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages pysam

RUN mkdir -p /data/genomics /solution

COPY data/genomics/sample_A.bam /data/genomics/sample_A.bam
COPY data/genomics/sample_A.bam.bai /data/genomics/sample_A.bam.bai
COPY data/genomics/sample_B.bam /data/genomics/sample_B.bam
COPY data/genomics/sample_C.bam /data/genomics/sample_C.bam
COPY data/genomics/sample_C.bam.bai /data/genomics/sample_C.bam.bai
COPY data/genomics/sample_D.bam /data/genomics/sample_D.bam
COPY data/genomics/sample_D.bam.bai /data/genomics/sample_D.bam.bai
COPY data/genomics/notes.txt /data/genomics/notes.txt
COPY data/genomics/sample_E.bam /data/genomics/sample_E.bam
COPY data/genomics/sample_E.bam.bai /data/genomics/sample_E.bam.bai

WORKDIR /data/genomics

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
