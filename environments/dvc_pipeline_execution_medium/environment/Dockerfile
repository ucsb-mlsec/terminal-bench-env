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

RUN pip3 install --no-cache-dir --break-system-packages dvc pandas

RUN mkdir -p /workspace/data /workspace/scripts /workspace/.dvc

COPY data/raw_transactions.csv /workspace/data/raw_transactions.csv
COPY dvc.yaml /workspace/dvc.yaml
COPY scripts/clean_data.py /workspace/scripts/clean_data.py
COPY scripts/calculate_metrics.py /workspace/scripts/calculate_metrics.py
COPY scripts/generate_report.py /workspace/scripts/generate_report.py
COPY .dvc/.gitignore /workspace/.dvc/.gitignore
COPY .dvc/config /workspace/.dvc/config

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
