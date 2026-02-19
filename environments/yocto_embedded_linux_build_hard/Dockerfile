FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/build-project/tools /workspace/build-project/config

COPY tools/validate-config /workspace/build-project/tools/validate-config
COPY config/base-system.conf /workspace/build-project/config/base-system.conf
COPY config/networking.conf /workspace/build-project/config/networking.conf
COPY config/utilities.conf /workspace/build-project/config/utilities.conf
COPY config/development.conf /workspace/build-project/config/development.conf

RUN chmod +x /workspace/build-project/tools/validate-config

WORKDIR /workspace