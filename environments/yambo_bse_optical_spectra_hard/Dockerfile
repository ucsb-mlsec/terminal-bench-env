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
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages numpy scipy

RUN mkdir -p /data/bands_80_kpoints_32_cutoff_6.0 && \
    mkdir -p /data/bands_100_kpoints_48_cutoff_7.0 && \
    mkdir -p /data/bands_120_kpoints_64_cutoff_8.0 && \
    mkdir -p /data/bands_140_kpoints_80_cutoff_9.0 && \
    mkdir -p /data/bands_160_kpoints_96_cutoff_10.0 && \
    mkdir -p /data/bands_120_kpoints_48_cutoff_8.0 && \
    mkdir -p /data/bands_120_kpoints_64_cutoff_6.0 && \
    mkdir -p /data/bands_100_kpoints_64_cutoff_8.0 && \
    mkdir -p /results

COPY data/bands_80_kpoints_32_cutoff_6.0/absorption_spectrum.dat /data/bands_80_kpoints_32_cutoff_6.0/absorption_spectrum.dat
COPY data/bands_100_kpoints_48_cutoff_7.0/absorption_spectrum.dat /data/bands_100_kpoints_48_cutoff_7.0/absorption_spectrum.dat
COPY data/bands_120_kpoints_64_cutoff_8.0/absorption_spectrum.dat /data/bands_120_kpoints_64_cutoff_8.0/absorption_spectrum.dat
COPY data/bands_140_kpoints_80_cutoff_9.0/absorption_spectrum.dat /data/bands_140_kpoints_80_cutoff_9.0/absorption_spectrum.dat
COPY data/bands_160_kpoints_96_cutoff_10.0/absorption_spectrum.dat /data/bands_160_kpoints_96_cutoff_10.0/absorption_spectrum.dat
COPY data/bands_120_kpoints_48_cutoff_8.0/absorption_spectrum.dat /data/bands_120_kpoints_48_cutoff_8.0/absorption_spectrum.dat
COPY data/bands_120_kpoints_64_cutoff_6.0/absorption_spectrum.dat /data/bands_120_kpoints_64_cutoff_6.0/absorption_spectrum.dat
COPY data/bands_100_kpoints_64_cutoff_8.0/absorption_spectrum.dat /data/bands_100_kpoints_64_cutoff_8.0/absorption_spectrum.dat

WORKDIR /workspace