FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/data_pipeline /tmp

COPY opt/data_pipeline/processor.py /opt/data_pipeline/processor.py
COPY opt/data_pipeline/test_pipeline.sh /opt/data_pipeline/test_pipeline.sh

RUN chmod +x /opt/data_pipeline/test_pipeline.sh

WORKDIR /opt/data_pipeline