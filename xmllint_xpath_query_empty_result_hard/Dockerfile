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
    apt-get install -y libxml2-utils && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/data_pipeline/configs

COPY data/vendor_a.xml /opt/data_pipeline/configs/vendor_a.xml
COPY data/vendor_b.xml /opt/data_pipeline/configs/vendor_b.xml
COPY data/vendor_c.xml /opt/data_pipeline/configs/vendor_c.xml
COPY data/vendor_d.xml /opt/data_pipeline/configs/vendor_d.xml
COPY data/vendor_e.xml /opt/data_pipeline/configs/vendor_e.xml
COPY extract.sh /opt/data_pipeline/extract.sh

RUN chmod +x /opt/data_pipeline/extract.sh

WORKDIR /opt/data_pipeline