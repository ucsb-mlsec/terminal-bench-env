FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    default-jre \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages stormpy

RUN wget -q https://www.prismmodelchecker.org/dl/prism-4.8-linux64-x86.tar.gz && \
    tar -xzf prism-4.8-linux64-x86.tar.gz && \
    mv prism-4.8-linux64-x86 /opt/prism && \
    rm prism-4.8-linux64-x86.tar.gz

ENV PATH="/opt/prism/bin:${PATH}"

COPY protocol.prism /workspace/protocol.prism

WORKDIR /workspace