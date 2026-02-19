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
RUN mkdir -p /home/user/solution \
    && mkdir -p /workspace/data \
    && mkdir -p /workspace/config

# Copy generated files to appropriate locations
COPY data/sample_transactions.csv /workspace/data/sample_transactions.csv
COPY partial_ddl.sql /workspace/partial_ddl.sql
COPY config/snowpipe_template.yaml /workspace/config/snowpipe_template.yaml

# Set working directory
WORKDIR /home/user/solution