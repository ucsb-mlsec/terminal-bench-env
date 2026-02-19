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
    apt-get install -y python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages pandas pandera

RUN mkdir -p /workspace/data /workspace/pipeline/schemas /workspace/solution

COPY data/raw_transactions.csv /workspace/data/raw_transactions.csv
COPY pipeline/stage1_ingestion.py /workspace/pipeline/stage1_ingestion.py
COPY pipeline/stage2_enrichment.py /workspace/pipeline/stage2_enrichment.py
COPY pipeline/stage3_aggregation.py /workspace/pipeline/stage3_aggregation.py
COPY pipeline/schemas/__init__.py /workspace/pipeline/schemas/__init__.py
COPY pipeline/schemas/stage1_schema.py /workspace/pipeline/schemas/stage1_schema.py
COPY pipeline/schemas/stage2_schema.py /workspace/pipeline/schemas/stage2_schema.py
COPY pipeline/schemas/stage3_schema.py /workspace/pipeline/schemas/stage3_schema.py
COPY run_pipeline.py /workspace/run_pipeline.py

WORKDIR /workspace