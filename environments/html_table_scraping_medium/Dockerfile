FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/catalog /workspace/output

RUN pip3 install --no-cache-dir --break-system-packages beautifulsoup4

COPY data/catalog/products_page1.html /workspace/catalog/products_page1.html
COPY data/catalog/products_page2.html /workspace/catalog/products_page2.html
COPY data/catalog/products_page3.html /workspace/catalog/products_page3.html
COPY data/catalog/legacy_catalog.html /workspace/catalog/legacy_catalog.html
COPY data/catalog/index.html /workspace/catalog/index.html

WORKDIR /workspace