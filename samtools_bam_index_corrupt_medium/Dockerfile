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

RUN mkdir -p /data/alignments /solution

COPY data/alignments/sample_A.bam /data/alignments/sample_A.bam
COPY data/alignments/sample_A.bam.bai /data/alignments/sample_A.bam.bai
COPY data/alignments/sample_B.bam /data/alignments/sample_B.bam
COPY data/alignments/sample_C.bam /data/alignments/sample_C.bam
COPY data/alignments/sample_C.bam.bai /data/alignments/sample_C.bam.bai

WORKDIR /data/alignments