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

RUN pip3 install --no-cache-dir --break-system-packages dask distributed pandas

RUN mkdir -p /workspace/data_pipeline/input /workspace/solution

COPY data/data_001.csv /workspace/data_pipeline/input/data_001.csv
COPY data/data_002.csv /workspace/data_pipeline/input/data_002.csv
COPY data/data_003.csv /workspace/data_pipeline/input/data_003.csv
COPY data/data_004.csv /workspace/data_pipeline/input/data_004.csv
COPY data/data_005.csv /workspace/data_pipeline/input/data_005.csv
COPY data/data_006.csv /workspace/data_pipeline/input/data_006.csv
COPY data/data_007.csv /workspace/data_pipeline/input/data_007.csv
COPY data/data_008.csv /workspace/data_pipeline/input/data_008.csv
COPY data/data_009.csv /workspace/data_pipeline/input/data_009.csv
COPY data/data_010.csv /workspace/data_pipeline/input/data_010.csv
COPY processor.py /workspace/data_pipeline/processor.py

WORKDIR /workspace