FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/configs /tmp/solution /workspace/data

COPY config/valid1.properties /workspace/configs/valid1.properties
COPY config/valid2.properties /workspace/configs/valid2.properties
COPY config/valid3.properties /workspace/configs/valid3.properties
COPY config/valid4.properties /workspace/configs/valid4.properties
COPY config/invalid1.properties /workspace/configs/invalid1.properties
COPY config/invalid2.properties /workspace/configs/invalid2.properties
COPY config/invalid3.properties /workspace/configs/invalid3.properties
COPY config/invalid4.properties /workspace/configs/invalid4.properties
COPY config/invalid5.properties /workspace/configs/invalid5.properties
COPY config/invalid6.properties /workspace/configs/invalid6.properties
COPY config/invalid7.properties /workspace/configs/invalid7.properties
COPY data/reachability.spc /workspace/data/reachability.spc

WORKDIR /workspace