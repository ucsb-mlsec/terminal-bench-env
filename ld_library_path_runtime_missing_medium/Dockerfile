FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/dataproc/bin /opt/dataproc/data /usr/local/lib/dataproc

COPY opt/dataproc/bin/processor /opt/dataproc/bin/processor
COPY usr/local/lib/dataproc/libcustomdata.so.1 /usr/local/lib/dataproc/libcustomdata.so.1
COPY data/sample.dat /opt/dataproc/data/sample.dat

RUN chmod +x /opt/dataproc/bin/processor

WORKDIR /opt/dataproc