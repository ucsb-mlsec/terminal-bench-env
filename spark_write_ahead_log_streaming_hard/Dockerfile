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
    openjdk-11-jdk-headless \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/spark_config /solution

COPY config/spark-defaults.conf /workspace/spark_config/spark-defaults.conf
COPY config/streaming-app.conf /workspace/spark_config/streaming-app.conf
COPY config/checkpoint-setup.sh /workspace/spark_config/checkpoint-setup.sh

WORKDIR /workspace