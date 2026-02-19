FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /incident_data /solution

COPY data/application.conf /incident_data/application.conf
COPY data/metrics_log.txt /incident_data/metrics_log.txt
COPY data/failure_analysis.txt /incident_data/failure_analysis.txt
COPY data/cluster_capacity.txt /incident_data/cluster_capacity.txt

WORKDIR /workspace