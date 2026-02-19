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
    apt-get install -y \
    bedtools \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y bedtools && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/bed_files

COPY data/peaks_sample1.bed /workspace/bed_files/peaks_sample1.bed
COPY data/peaks_sample2.bed /workspace/bed_files/peaks_sample2.bed
COPY data/regions_control.bed /workspace/bed_files/regions_control.bed
COPY data/regions_test.bed /workspace/bed_files/regions_test.bed
COPY process_beds.sh /workspace/process_beds.sh

RUN chmod +x /workspace/process_beds.sh

WORKDIR /workspace