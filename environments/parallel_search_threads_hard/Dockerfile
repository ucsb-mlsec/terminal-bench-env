FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/genomes

COPY data/genomes/genome_01.txt /workspace/genomes/genome_01.txt
COPY data/genomes/genome_02.txt /workspace/genomes/genome_02.txt
COPY data/genomes/genome_03.txt /workspace/genomes/genome_03.txt
COPY data/genomes/genome_04.txt /workspace/genomes/genome_04.txt
COPY data/genomes/genome_05.txt /workspace/genomes/genome_05.txt
COPY data/genomes/genome_06.txt /workspace/genomes/genome_06.txt
COPY data/genomes/genome_07.txt /workspace/genomes/genome_07.txt
COPY data/genomes/genome_08.txt /workspace/genomes/genome_08.txt
COPY data/patterns.txt /workspace/patterns.txt

WORKDIR /workspace