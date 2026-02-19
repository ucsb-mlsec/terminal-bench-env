FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/sales_pipeline/data

RUN pip3 install --no-cache-dir --break-system-packages dagster dagster-webserver pandas

COPY sales_pipeline/data/transactions.csv /workspace/sales_pipeline/data/transactions.csv
COPY sales_pipeline/workspace.yaml /workspace/sales_pipeline/workspace.yaml
COPY sales_pipeline/__init__.py /workspace/sales_pipeline/__init__.py
COPY sales_pipeline/pyproject.toml /workspace/sales_pipeline/pyproject.toml

WORKDIR /workspace/sales_pipeline