FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/raw_invoices

COPY data/raw_invoices/vendor_a_001.txt /workspace/raw_invoices/vendor_a_001.txt
COPY data/raw_invoices/vendor_b_002.txt /workspace/raw_invoices/vendor_b_002.txt
COPY data/raw_invoices/vendor_c_003.txt /workspace/raw_invoices/vendor_c_003.txt
COPY data/raw_invoices/vendor_d_004.txt /workspace/raw_invoices/vendor_d_004.txt
COPY data/raw_invoices/vendor_e_005.txt /workspace/raw_invoices/vendor_e_005.txt
COPY data/raw_invoices/vendor_f_006.txt /workspace/raw_invoices/vendor_f_006.txt
COPY data/raw_invoices/vendor_g_007.txt /workspace/raw_invoices/vendor_g_007.txt
COPY data/raw_invoices/vendor_h_008.txt /workspace/raw_invoices/vendor_h_008.txt

WORKDIR /workspace