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
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages pypdf2 pdfplumber

RUN pip3 install --no-cache-dir --break-system-packages \
    PyPDF2 \
    pdfplumber \
    pypdf

RUN mkdir -p /workspace/invoices

COPY data/invoices/invoice_001.pdf /workspace/invoices/invoice_001.pdf
COPY data/invoices/invoice_002.pdf /workspace/invoices/invoice_002.pdf
COPY data/invoices/invoice_003.pdf /workspace/invoices/invoice_003.pdf

WORKDIR /workspace