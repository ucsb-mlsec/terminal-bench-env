FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN apt-get update && \
    apt-get install -y protobuf-compiler python3-protobuf && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages protobuf

RUN mkdir -p /app /app/samples /solution

COPY data/auth.proto /app/auth.proto
COPY data/field_types.txt /app/field_types.txt
COPY data/samples/auth_001.bin /app/samples/auth_001.bin
COPY data/samples/auth_002.bin /app/samples/auth_002.bin
COPY data/samples/auth_003.bin /app/samples/auth_003.bin
COPY data/samples/auth_004.bin /app/samples/auth_004.bin
COPY data/samples/auth_005.bin /app/samples/auth_005.bin
COPY data/samples/auth_006.bin /app/samples/auth_006.bin
COPY data/samples/auth_007.bin /app/samples/auth_007.bin
COPY data/samples/auth_008.bin /app/samples/auth_008.bin
COPY data/samples/auth_009.bin /app/samples/auth_009.bin
COPY data/samples/auth_010.bin /app/samples/auth_010.bin

WORKDIR /app