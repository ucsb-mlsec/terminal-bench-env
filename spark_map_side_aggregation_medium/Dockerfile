FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install system dependencies including Java
RUN apt-get update && \
    apt-get install -y python3-pip openjdk-11-jdk-headless wget && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1 pyspark==3.5.0

# Create necessary directories
RUN mkdir -p /workspace/spark_app/data /workspace/solution

# Copy application files
COPY data/orders.csv /workspace/spark_app/data/orders.csv
COPY config/spark.conf /workspace/spark_app/spark.conf

# Set Spark environment variables
ENV SPARK_HOME=/usr/local/lib/python3.12/dist-packages/pyspark
ENV PATH=$PATH:$SPARK_HOME/bin
ENV PYSPARK_PYTHON=python3

WORKDIR /workspace