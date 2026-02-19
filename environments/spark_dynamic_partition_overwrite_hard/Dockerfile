FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install Java (required for PySpark) and other dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openjdk-11-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y openjdk-11-jre-headless && \
    rm -rf /var/lib/apt/lists/*

# Install PySpark
RUN pip3 install --no-cache-dir --break-system-packages pyspark

# Create necessary directories
RUN mkdir -p /workspace/input /workspace/output/sales /workspace/data/sales

# Copy the PySpark application
COPY sales_pipeline.py /workspace/sales_pipeline.py

# Copy input data
COPY data/sales_update.csv /workspace/input/sales_update.csv

# Copy existing partitioned sales data
COPY data/sales/date=2024-01-15/store=Chicago/part-00000.csv /workspace/output/sales/date=2024-01-15/store=Chicago/part-00000.csv
COPY data/sales/date=2024-01-15/store=NewYork/part-00000.csv /workspace/output/sales/date=2024-01-15/store=NewYork/part-00000.csv
COPY data/sales/date=2024-01-15/store=LosAngeles/part-00000.csv /workspace/output/sales/date=2024-01-15/store=LosAngeles/part-00000.csv
COPY data/sales/date=2024-01-16/store=Chicago/part-00000.csv /workspace/output/sales/date=2024-01-16/store=Chicago/part-00000.csv
COPY data/sales/date=2024-01-16/store=NewYork/part-00000.csv /workspace/output/sales/date=2024-01-16/store=NewYork/part-00000.csv

# Set working directory
WORKDIR /workspace

# Set JAVA_HOME environment variable
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64