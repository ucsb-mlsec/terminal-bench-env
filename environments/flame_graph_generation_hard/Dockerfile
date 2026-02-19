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
    python3 \
    python3-pip \
    linux-tools-generic \
    linux-tools-common \
    perl \
    git && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y linux-tools-generic linux-tools-common && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages py-spy

RUN git clone https://github.com/brendangregg/FlameGraph.git /opt/FlameGraph

RUN mkdir -p /opt/logprocessor/data

COPY opt/logprocessor/processor.py /opt/logprocessor/processor.py
COPY data/access.log /opt/logprocessor/data/access.log

WORKDIR /opt/logprocessor