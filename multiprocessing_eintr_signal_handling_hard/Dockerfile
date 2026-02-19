FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/dataprocessor/data /solution

COPY processor.py /opt/dataprocessor/processor.py
COPY data/dataset_001.csv /opt/dataprocessor/data/dataset_001.csv
COPY data/dataset_002.csv /opt/dataprocessor/data/dataset_002.csv
COPY data/dataset_003.csv /opt/dataprocessor/data/dataset_003.csv
COPY data/dataset_004.csv /opt/dataprocessor/data/dataset_004.csv
COPY test_processor.sh /opt/dataprocessor/test_processor.sh

RUN chmod +x /opt/dataprocessor/test_processor.sh

WORKDIR /opt/dataprocessor