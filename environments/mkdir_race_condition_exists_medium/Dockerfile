FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Create necessary directories
RUN mkdir -p /opt/logprocessor \
    /var/logs/incoming \
    /var/logs/processed \
    /opt/logprocessor/data

# Copy all generated files to the container
COPY process_logs.sh /opt/logprocessor/process_logs.sh
COPY run_parallel_test.sh /opt/logprocessor/run_parallel_test.sh
COPY data/sample_01.log /opt/logprocessor/data/sample_01.log
COPY data/sample_02.log /opt/logprocessor/data/sample_02.log

# Make scripts executable
RUN chmod +x /opt/logprocessor/process_logs.sh \
    /opt/logprocessor/run_parallel_test.sh

# Set working directory
WORKDIR /opt/logprocessor