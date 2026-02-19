FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/data_pipeline/input \
    /opt/data_pipeline/scripts \
    /opt/data_pipeline/logs \
    /opt/data_pipeline/solution

COPY data/customers.csv /opt/data_pipeline/input/customers.csv
COPY data/transactions.csv /opt/data_pipeline/input/transactions.csv
COPY data/products.csv /opt/data_pipeline/input/products.csv
COPY join_data.sh /opt/data_pipeline/scripts/join_data.sh
COPY data/pipeline.log /opt/data_pipeline/logs/pipeline.log

RUN chmod +x /opt/data_pipeline/scripts/join_data.sh

WORKDIR /opt/data_pipeline