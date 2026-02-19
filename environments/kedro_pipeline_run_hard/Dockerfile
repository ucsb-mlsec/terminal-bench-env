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
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages pyyaml

RUN pip3 install --no-cache-dir --break-system-packages kedro pandas

RUN mkdir -p /home/agent/sales_pipeline/data \
    /home/agent/sales_pipeline/conf/base \
    /home/agent/sales_pipeline/src/sales_pipeline/pipelines/data_processing \
    /home/agent

COPY data/sales_data.csv /home/agent/sales_pipeline/data/sales_data.csv
COPY sales_pipeline/conf/base/catalog.yml /home/agent/sales_pipeline/conf/base/catalog.yml
COPY sales_pipeline/conf/base/parameters.yml /home/agent/sales_pipeline/conf/base/parameters.yml
COPY sales_pipeline/src/sales_pipeline/pipelines/data_processing/nodes.py /home/agent/sales_pipeline/src/sales_pipeline/pipelines/data_processing/nodes.py
COPY sales_pipeline/src/sales_pipeline/pipelines/data_processing/pipeline.py /home/agent/sales_pipeline/src/sales_pipeline/pipelines/data_processing/pipeline.py
COPY sales_pipeline/src/sales_pipeline/pipelines/data_processing/__init__.py /home/agent/sales_pipeline/src/sales_pipeline/pipelines/data_processing/__init__.py
COPY sales_pipeline/src/sales_pipeline/pipeline_registry.py /home/agent/sales_pipeline/src/sales_pipeline/pipeline_registry.py
COPY sales_pipeline/src/sales_pipeline/__init__.py /home/agent/sales_pipeline/src/sales_pipeline/__init__.py
COPY sales_pipeline/src/sales_pipeline/settings.py /home/agent/sales_pipeline/src/sales_pipeline/settings.py
COPY sales_pipeline/pyproject.toml /home/agent/sales_pipeline/pyproject.toml
COPY sales_pipeline/.kedro.yml /home/agent/sales_pipeline/.kedro.yml

WORKDIR /home/agent/sales_pipeline