FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y bash && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /data/customer_export /data/auth /workspace

COPY data/customer_export/page_001.json /data/customer_export/page_001.json
COPY data/customer_export/page_002.json /data/customer_export/page_002.json
COPY data/customer_export/page_003.json /data/customer_export/page_003.json
COPY data/customer_export/page_004.json /data/customer_export/page_004.json
COPY data/customer_export/page_005.json /data/customer_export/page_005.json
COPY data/customer_export/metadata.json /data/customer_export/metadata.json
COPY data/auth/refresh_token.sh /data/auth/refresh_token.sh
COPY extract_customers.py /workspace/extract_customers.py

RUN chmod +x /data/auth/refresh_token.sh

WORKDIR /workspace