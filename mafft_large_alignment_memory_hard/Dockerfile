FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /data/incoming /solution

COPY data/incoming/sample_001.fasta /data/incoming/sample_001.fasta
COPY data/incoming/sample_002.fasta /data/incoming/sample_002.fasta
COPY data/incoming/sample_003.fasta /data/incoming/sample_003.fasta
COPY data/incoming/sample_004.fasta /data/incoming/sample_004.fasta
COPY data/incoming/sample_005.fasta /data/incoming/sample_005.fasta
COPY data/incoming/sample_006.fasta /data/incoming/sample_006.fasta
COPY data/incoming/sample_007.fasta /data/incoming/sample_007.fasta
COPY data/incoming/sample_008.fasta /data/incoming/sample_008.fasta
COPY data/incoming/sample_009.fasta /data/incoming/sample_009.fasta
COPY data/incoming/sample_010.fasta /data/incoming/sample_010.fasta
COPY data/incoming/sample_011.fasta /data/incoming/sample_011.fasta
COPY data/incoming/sample_012.fasta /data/incoming/sample_012.fasta
COPY data/incoming/sample_013.fasta /data/incoming/sample_013.fasta
COPY data/incoming/sample_014.fasta /data/incoming/sample_014.fasta
COPY data/incoming/sample_015.fasta /data/incoming/sample_015.fasta

WORKDIR /workspace