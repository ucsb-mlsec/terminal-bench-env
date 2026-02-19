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
    wine \
    wine64 \
    wget \
    cabextract && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y wine cabextract && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/dxsdk && \
    cd /opt/dxsdk && \
    wget -q https://download.microsoft.com/download/A/E/7/AE743F1F-632B-4809-87A9-AA1BB3458E31/DXSDK_Jun10.exe || true

RUN mkdir -p /workspace/shader

COPY shader/broken_pixel.hlsl /workspace/shader/broken_pixel.hlsl
COPY shader/vertex.hlsl /workspace/shader/vertex.hlsl
COPY compiler_output.txt /workspace/compiler_output.txt

WORKDIR /workspace