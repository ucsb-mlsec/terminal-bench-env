FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/worker_app/data /opt/worker_app/output

COPY process_data.py /opt/worker_app/process_data.py
COPY data/file1.csv /opt/worker_app/data/file1.csv
COPY data/file2.csv /opt/worker_app/data/file2.csv
COPY data/file3.csv /opt/worker_app/data/file3.csv
COPY data/file4.csv /opt/worker_app/data/file4.csv

WORKDIR /opt/worker_app