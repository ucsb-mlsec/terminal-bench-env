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
    apt-get install -y systemd && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/systemd/system \
    /opt/dataproc/bin \
    /var/lib/dataproc \
    /var/log/dataproc \
    /root

RUN useradd -r -s /bin/false dataproc || true

RUN touch /opt/dataproc/bin/processor && \
    chmod +x /opt/dataproc/bin/processor && \
    chown -R dataproc:dataproc /var/lib/dataproc /var/log/dataproc

COPY config/data-processor.service /etc/systemd/system/data-processor.service

WORKDIR /root