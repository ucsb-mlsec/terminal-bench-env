FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y binutils && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/production/bin /opt/production/lib

COPY opt/production/bin/processor /opt/production/bin/processor
COPY opt/production/lib/libcustom1.so.1 /opt/production/lib/libcustom1.so.1
COPY opt/production/lib/libcustom2.so.1 /opt/production/lib/libcustom2.so.1

RUN chmod +x /opt/production/bin/processor

WORKDIR /workspace