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
    apt-get install -y \
    binutils \
    file \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y binutils gcc libc6-dev && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/libcalc-1.0 /workspace/libcalc-2.0 /home/agent

COPY libcalc-1.0/calc.h /workspace/libcalc-1.0/calc.h
COPY libcalc-1.0/libcalc.so /workspace/libcalc-1.0/libcalc.so
COPY libcalc-2.0/calc.h /workspace/libcalc-2.0/calc.h
COPY libcalc-2.0/libcalc.so /workspace/libcalc-2.0/libcalc.so

WORKDIR /workspace