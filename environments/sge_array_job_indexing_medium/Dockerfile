FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /data/sequences /jobs /solution

COPY data/sequences/sample_001.fasta /data/sequences/sample_001.fasta
COPY data/sequences/sample_002.fasta /data/sequences/sample_002.fasta
COPY data/sequences/sample_003.fasta /data/sequences/sample_003.fasta
COPY data/sequences/sample_050.fasta /data/sequences/sample_050.fasta
COPY jobs/process_sequences.sh /jobs/process_sequences.sh
COPY jobs/task_mapping.conf /jobs/task_mapping.conf

RUN chmod +x /jobs/process_sequences.sh

WORKDIR /solution