FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gdb strace ltrace binutils && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /usr/local/bin /usr/local/lib /home/agent

COPY usr/local/bin/license_validator /usr/local/bin/license_validator
COPY usr/local/lib/liblicense_check.so /usr/local/lib/liblicense_check.so

RUN chmod +x /usr/local/bin/license_validator && \
    chmod 644 /usr/local/lib/liblicense_check.so && \
    ldconfig

WORKDIR /home/agent