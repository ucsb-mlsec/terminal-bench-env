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
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages biopython

RUN mkdir -p /data /solution

COPY data/species_a_contigs.txt /data/species_a_contigs.txt
COPY data/species_b_assembly.gff /data/species_b_assembly.gff
COPY data/species_c_raw.fasta /data/species_c_raw.fasta
COPY data/color_scheme.conf /data/color_scheme.conf
COPY data/reference_bands.txt /data/reference_bands.txt

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
