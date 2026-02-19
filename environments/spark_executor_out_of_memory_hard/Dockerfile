FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y python3-pip openjdk-11-jdk-headless wget && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages pyspark

# Install Python dependencies
RUN pip3 install --no-cache-dir --break-system-packages \
    pytest==8.4.1 \
    pandas==2.3.2 \
    scipy==1.16.1 \
    pyspark==3.4.0 \
    pyarrow

# Create directory structure
RUN mkdir -p /opt/spark/jobs \
    /opt/spark/conf \
    /data/transactions \
    /var/log/spark/executor-failures \
    /solution

# Copy application files
COPY transaction_aggregator.py /opt/spark/jobs/transaction_aggregator.py
COPY config/spark-defaults.conf /opt/spark/conf/spark-defaults.conf
COPY data/executor_oom_error_1.log /var/log/spark/executor-failures/executor_oom_error_1.log
COPY data/executor_oom_error_2.log /var/log/spark/executor-failures/executor_oom_error_2.log
COPY data/executor_oom_error_3.log /var/log/spark/executor-failures/executor_oom_error_3.log
COPY data/sample_transactions.parquet /data/transactions/sample_transactions.parquet

# Set environment variables
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
ENV PYSPARK_PYTHON=python3

WORKDIR /workspace