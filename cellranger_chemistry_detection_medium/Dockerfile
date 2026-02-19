FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /data/fastq /solution

COPY data/chemistry_specs.txt /data/chemistry_specs.txt
COPY data/README.txt /data/README.txt
COPY data/fastq/sample001_R1.fastq /data/fastq/sample001_R1.fastq
COPY data/fastq/sample001_R2.fastq /data/fastq/sample001_R2.fastq
COPY data/fastq/sample002_R1.fastq /data/fastq/sample002_R1.fastq
COPY data/fastq/sample002_R2.fastq /data/fastq/sample002_R2.fastq

WORKDIR /workspace