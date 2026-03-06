FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages pyspark

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/spark-app/config /solution

COPY config/spark-defaults.conf /opt/spark-app/config/spark-defaults.conf
COPY config/application.properties /opt/spark-app/config/application.properties
COPY config/spark.conf /opt/spark-app/config/spark.conf
COPY config/cluster.yaml /opt/spark-app/config/cluster.yaml
COPY config/legacy-spark.properties /opt/spark-app/config/legacy-spark.properties
COPY process_transactions.py /opt/spark-app/process_transactions.py
COPY README.txt /opt/spark-app/README.txt

WORKDIR /opt/spark-app

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
