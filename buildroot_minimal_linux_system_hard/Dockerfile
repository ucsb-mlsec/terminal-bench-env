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
    apt-get install -y --no-install-recommends \
    gcc \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc make && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/buildsys/package \
    /workspace/buildsys/configs \
    /workspace/app-source \
    /tmp

COPY buildsys/Config.in /workspace/buildsys/Config.in
COPY buildsys/configs/minimal_defconfig /workspace/buildsys/configs/minimal_defconfig
COPY buildsys/validate_build.sh /workspace/buildsys/validate_build.sh
COPY buildsys/package/README.txt /workspace/buildsys/package/README.txt
COPY app-source/monitor.c /workspace/app-source/monitor.c
COPY README.txt /workspace/README.txt

RUN chmod +x /workspace/buildsys/validate_build.sh

WORKDIR /workspace