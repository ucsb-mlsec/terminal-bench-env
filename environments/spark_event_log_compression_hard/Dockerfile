FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /task_data /solution /var/spark/logs /var/spark/eventLog

COPY data/requirements.txt /task_data/requirements.txt
COPY config/spark-defaults-prod.conf /task_data/spark-defaults-prod.conf
COPY config/spark-defaults-dev.conf /task_data/spark-defaults-dev.conf
COPY config/history-server.properties /task_data/history-server.properties

WORKDIR /workspace