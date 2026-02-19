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
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages pyyaml

RUN pip3 install --no-cache-dir --break-system-packages dvc pandas

WORKDIR /workspace

RUN mkdir -p /workspace/data

COPY data/sales_transactions.csv /workspace/data/sales_transactions.csv
COPY dvc.yaml /workspace/dvc.yaml
COPY params.yaml /workspace/params.yaml
COPY validate_data.py /workspace/validate_data.py
COPY generate_report.py /workspace/generate_report.py