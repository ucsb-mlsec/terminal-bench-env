FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /tmp/smart_reports

COPY data/smart_reports/sda.txt /tmp/smart_reports/sda.txt
COPY data/smart_reports/sdb.txt /tmp/smart_reports/sdb.txt
COPY data/smart_reports/sdc.txt /tmp/smart_reports/sdc.txt
COPY data/smart_reports/sdd.txt /tmp/smart_reports/sdd.txt
COPY data/smart_reports/sde.txt /tmp/smart_reports/sde.txt

WORKDIR /workspace