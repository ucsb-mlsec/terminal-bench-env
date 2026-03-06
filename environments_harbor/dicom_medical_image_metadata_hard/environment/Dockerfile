FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /data/dicom_metadata /solution

COPY data/dicom_metadata/scan_001.json /data/dicom_metadata/scan_001.json
COPY data/dicom_metadata/scan_002.json /data/dicom_metadata/scan_002.json
COPY data/dicom_metadata/scan_003.json /data/dicom_metadata/scan_003.json
COPY data/dicom_metadata/scan_004.json /data/dicom_metadata/scan_004.json
COPY data/dicom_metadata/scan_005.json /data/dicom_metadata/scan_005.json
COPY data/dicom_metadata/scan_006.json /data/dicom_metadata/scan_006.json
COPY data/dicom_metadata/scan_007.json /data/dicom_metadata/scan_007.json
COPY data/dicom_metadata/scan_008.json /data/dicom_metadata/scan_008.json
COPY data/dicom_metadata/scan_009.json /data/dicom_metadata/scan_009.json
COPY data/dicom_metadata/scan_010.json /data/dicom_metadata/scan_010.json
COPY data/dicom_metadata/scan_011.json /data/dicom_metadata/scan_011.json
COPY data/dicom_metadata/scan_012.json /data/dicom_metadata/scan_012.json
COPY data/dicom_metadata/scan_014.json /data/dicom_metadata/scan_014.json
COPY data/dicom_metadata/scan_015.json /data/dicom_metadata/scan_015.json

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
