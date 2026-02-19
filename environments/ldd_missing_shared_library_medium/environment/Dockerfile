FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install additional tools
RUN apt-get update && \
    apt-get install -y \
    binutils \
    file \
    && rm -rf /var/lib/apt/lists/*

# Create directories
RUN mkdir -p /usr/local/bin /opt/customlibs

# Copy binary and libraries
COPY usr/local/bin/dataprocessor /usr/local/bin/dataprocessor
COPY opt/customlibs/libcustom1.so.1 /opt/customlibs/libcustom1.so.1
COPY opt/customlibs/libcustom2.so.1 /opt/customlibs/libcustom2.so.1
COPY opt/customlibs/libcustom3.so.1 /opt/customlibs/libcustom3.so.1

# Set execute permissions
RUN chmod +x /usr/local/bin/dataprocessor

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
