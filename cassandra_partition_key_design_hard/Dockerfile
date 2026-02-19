FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install Cassandra and dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    gnupg \
    apt-transport-https \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /opt/sensor_system

# Copy all required files to the container
COPY data/current_schema.cql /opt/sensor_system/current_schema.cql
COPY data/query_workload.txt /opt/sensor_system/query_workload.txt
COPY data/performance_requirements.txt /opt/sensor_system/performance_requirements.txt
COPY data/sample_data.txt /opt/sensor_system/sample_data.txt

# Set working directory
WORKDIR /opt/sensor_system