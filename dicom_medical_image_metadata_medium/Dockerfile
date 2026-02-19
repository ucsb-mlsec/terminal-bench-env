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

RUN pip3 install --no-cache-dir --break-system-packages pydicom

RUN mkdir -p /medical_data/scans /output

COPY data/scans/patient001_scan1.dcm /medical_data/scans/patient001_scan1.dcm
COPY data/scans/patient001_scan2.dcm /medical_data/scans/patient001_scan2.dcm
COPY data/scans/patient002_scan1.dcm /medical_data/scans/patient002_scan1.dcm
COPY data/scans/patient002_scan2.dcm /medical_data/scans/patient002_scan2.dcm
COPY data/scans/patient003_scan1.dcm /medical_data/scans/patient003_scan1.dcm
COPY data/scans/patient004_scan1.dcm /medical_data/scans/patient004_scan1.dcm
COPY data/scans/patient004_scan2.dcm /medical_data/scans/patient004_scan2.dcm
COPY data/scans/patient005_scan1.dcm /medical_data/scans/patient005_scan1.dcm

WORKDIR /workspace