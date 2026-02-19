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
    apt-get install -y build-essential gcc make binutils && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/embedded_app/build

COPY embedded_app/main.c /workspace/embedded_app/main.c
COPY embedded_app/utils.c /workspace/embedded_app/utils.c
COPY embedded_app/data_tables.c /workspace/embedded_app/data_tables.c
COPY embedded_app/Makefile /workspace/embedded_app/Makefile

WORKDIR /workspace/embedded_app