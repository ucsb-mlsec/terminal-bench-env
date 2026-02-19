FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/invoice_texts

COPY data/invoice_texts/invoice_001.txt /workspace/invoice_texts/invoice_001.txt
COPY data/invoice_texts/invoice_002.txt /workspace/invoice_texts/invoice_002.txt
COPY data/invoice_texts/invoice_003.txt /workspace/invoice_texts/invoice_003.txt
COPY data/invoice_texts/invoice_004.txt /workspace/invoice_texts/invoice_004.txt
COPY data/invoice_texts/invoice_005.txt /workspace/invoice_texts/invoice_005.txt
COPY data/invoice_texts/invoice_006.txt /workspace/invoice_texts/invoice_006.txt
COPY data/invoice_texts/invoice_007.txt /workspace/invoice_texts/invoice_007.txt
COPY data/invoice_texts/invoice_008.txt /workspace/invoice_texts/invoice_008.txt
COPY data/invoice_texts/invoice_009.txt /workspace/invoice_texts/invoice_009.txt
COPY data/invoice_texts/invoice_010.txt /workspace/invoice_texts/invoice_010.txt
COPY data/invoice_texts/invoice_011.txt /workspace/invoice_texts/invoice_011.txt
COPY data/invoice_texts/invoice_012.txt /workspace/invoice_texts/invoice_012.txt

WORKDIR /workspace