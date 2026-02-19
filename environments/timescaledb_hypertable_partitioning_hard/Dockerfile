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
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages pandas

RUN mkdir -p /home/user/data /home/user/output

COPY data/sensors_2023.csv /home/user/data/sensors_2023.csv
COPY data/sensors_2024_q1.csv /home/user/data/sensors_2024_q1.csv
COPY data/misc_readings.csv /home/user/data/misc_readings.csv

WORKDIR /home/user