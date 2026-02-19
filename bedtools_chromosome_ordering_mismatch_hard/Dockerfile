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

RUN mkdir -p /data/bed_files /data/reference /data/corrected /solution

COPY data/bed_files/facility_a_sample1.bed /data/bed_files/facility_a_sample1.bed
COPY data/bed_files/facility_a_sample2.bed /data/bed_files/facility_a_sample2.bed
COPY data/bed_files/facility_b_sample1.bed /data/bed_files/facility_b_sample1.bed
COPY data/bed_files/facility_b_sample2.bed /data/bed_files/facility_b_sample2.bed
COPY data/bed_files/facility_c_sample1.bed /data/bed_files/facility_c_sample1.bed
COPY data/bed_files/facility_c_sample2.bed /data/bed_files/facility_c_sample2.bed
COPY data/reference/hg38.chrom.sizes /data/reference/hg38.chrom.sizes

WORKDIR /solution