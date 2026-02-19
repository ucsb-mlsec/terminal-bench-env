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
    file \
    libmagic-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y file && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/samples /workspace/solution

COPY data/qdat_v1_temp.qdat /workspace/samples/qdat_v1_temp.qdat
COPY data/qdat_v1_pres.qdat /workspace/samples/qdat_v1_pres.qdat
COPY data/qdat_v1_magn.qdat /workspace/samples/qdat_v1_magn.qdat
COPY data/qdat_v2_temp.qdat /workspace/samples/qdat_v2_temp.qdat
COPY data/qdat_v2_pres.qdat /workspace/samples/qdat_v2_pres.qdat
COPY data/qdat_v2_magn.qdat /workspace/samples/qdat_v2_magn.qdat
COPY data/qdat_v3_temp.qdat /workspace/samples/qdat_v3_temp.qdat
COPY data/qdat_v3_pres.qdat /workspace/samples/qdat_v3_pres.qdat
COPY data/qdat_v3_magn.qdat /workspace/samples/qdat_v3_magn.qdat

WORKDIR /workspace