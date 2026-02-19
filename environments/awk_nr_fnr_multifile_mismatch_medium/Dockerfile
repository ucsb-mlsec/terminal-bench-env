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
    apt-get install -y gawk && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /data/sales /scripts /expected /solution

COPY data/sales/region_north.csv /data/sales/region_north.csv
COPY data/sales/region_south.csv /data/sales/region_south.csv
COPY data/sales/region_west.csv /data/sales/region_west.csv
COPY scripts/process_sales.awk /scripts/process_sales.awk
COPY expected/correct_output.txt /expected/correct_output.txt

WORKDIR /workspace