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
    python3-pip \
    chromium-browser \
    chromium-chromedriver && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y chromium chromium-driver && \
    pip3 install --no-cache-dir --break-system-packages flask selenium beautifulsoup4 && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages \
    flask \
    selenium

RUN mkdir -p /workspace/catalog_app

COPY catalog_app/server.py /workspace/catalog_app/server.py
COPY catalog_app/index.html /workspace/catalog_app/index.html

WORKDIR /workspace