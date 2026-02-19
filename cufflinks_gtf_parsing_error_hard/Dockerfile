FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /data/gtf_files /results

COPY data/gtf_files/sample1.gtf /data/gtf_files/sample1.gtf
COPY data/gtf_files/sample2.gtf /data/gtf_files/sample2.gtf
COPY data/gtf_files/sample3.gtf /data/gtf_files/sample3.gtf
COPY data/gtf_files/sample4.gtf /data/gtf_files/sample4.gtf
COPY data/gtf_files/sample5.gtf /data/gtf_files/sample5.gtf
COPY data/gtf_files/sample6.gtf /data/gtf_files/sample6.gtf
COPY data/gtf_files/sample7.gtf /data/gtf_files/sample7.gtf
COPY data/gtf_files/sample8.gtf /data/gtf_files/sample8.gtf
COPY data/gtf_files/valid_complete.gtf /data/gtf_files/valid_complete.gtf
COPY data/gtf_files/corrupted_mixed.gtf /data/gtf_files/corrupted_mixed.gtf

WORKDIR /workspace