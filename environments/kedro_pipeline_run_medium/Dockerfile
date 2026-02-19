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
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages kedro pandas

RUN mkdir -p /home/project/sales_pipeline/data/01_raw \
    /home/project/sales_pipeline/conf/base \
    /home/project/sales_pipeline/src/sales_pipeline/pipelines/data_processing

COPY data/sales.csv /home/project/sales_pipeline/data/01_raw/sales.csv
COPY config/catalog.yml /home/project/sales_pipeline/conf/base/catalog.yml
COPY src/sales_pipeline/pipelines/data_processing/nodes.py /home/project/sales_pipeline/src/sales_pipeline/pipelines/data_processing/nodes.py
COPY src/sales_pipeline/pipelines/data_processing/pipeline.py /home/project/sales_pipeline/src/sales_pipeline/pipelines/data_processing/pipeline.py

WORKDIR /home/project/sales_pipeline