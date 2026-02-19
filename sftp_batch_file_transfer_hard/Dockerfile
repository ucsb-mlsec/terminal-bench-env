FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /home/agent/sftp_staging/incoming /home/agent

COPY data/region_north.csv /home/agent/sftp_staging/incoming/region_north.csv
COPY data/region_south.csv /home/agent/sftp_staging/incoming/region_south.csv
COPY data/region_east.csv /home/agent/sftp_staging/incoming/region_east.csv
COPY data/region_west.csv /home/agent/sftp_staging/incoming/region_west.csv
COPY data/region_central.csv /home/agent/sftp_staging/incoming/region_central.csv

WORKDIR /home/agent