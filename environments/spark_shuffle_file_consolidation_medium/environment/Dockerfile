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
    apt-get install -y --no-install-recommends \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/spark/shuffle_data /opt/spark/conf /opt/spark/logs /solution

COPY config/spark-defaults.conf /opt/spark/conf/spark-defaults.conf
COPY data/application.log /opt/spark/logs/application.log
COPY data/shuffle_data_structure.txt /opt/spark/shuffle_data/shuffle_data_structure.txt

WORKDIR /opt/spark

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
