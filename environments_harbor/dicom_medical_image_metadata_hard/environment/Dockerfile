FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/data/dicom_metadata /workspace/solution

COPY data/dicom_metadata/scan_001.json /workspace/data/dicom_metadata/scan_001.json
COPY data/dicom_metadata/scan_002.json /workspace/data/dicom_metadata/scan_002.json
COPY data/dicom_metadata/scan_003.json /workspace/data/dicom_metadata/scan_003.json
COPY data/dicom_metadata/scan_004.json /workspace/data/dicom_metadata/scan_004.json
COPY data/dicom_metadata/scan_005.json /workspace/data/dicom_metadata/scan_005.json
COPY data/dicom_metadata/scan_006.json /workspace/data/dicom_metadata/scan_006.json
COPY data/dicom_metadata/scan_007.json /workspace/data/dicom_metadata/scan_007.json
COPY data/dicom_metadata/scan_008.json /workspace/data/dicom_metadata/scan_008.json
COPY data/dicom_metadata/scan_009.json /workspace/data/dicom_metadata/scan_009.json
COPY data/dicom_metadata/scan_010.json /workspace/data/dicom_metadata/scan_010.json
COPY data/dicom_metadata/scan_011.json /workspace/data/dicom_metadata/scan_011.json
COPY data/dicom_metadata/scan_012.json /workspace/data/dicom_metadata/scan_012.json
COPY data/dicom_metadata/scan_014.json /workspace/data/dicom_metadata/scan_014.json
COPY data/dicom_metadata/scan_015.json /workspace/data/dicom_metadata/scan_015.json

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
