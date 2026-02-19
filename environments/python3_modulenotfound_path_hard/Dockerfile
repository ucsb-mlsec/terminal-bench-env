FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/research/libs/scicore \
    /usr/local/research/dataproc \
    /home/research/analysis \
    /solution

COPY opt/research/libs/scicore/__init__.py /opt/research/libs/scicore/__init__.py
COPY opt/research/libs/scicore/math_utils.py /opt/research/libs/scicore/math_utils.py
COPY opt/research/libs/scicore/data_ops.py /opt/research/libs/scicore/data_ops.py
COPY usr/local/research/dataproc/pipeline.py /usr/local/research/dataproc/pipeline.py
COPY usr/local/research/dataproc/transformers.py /usr/local/research/dataproc/transformers.py
COPY home/research/analysis/__init__.py /home/research/analysis/__init__.py
COPY home/research/analysis/compute.py /home/research/analysis/compute.py

WORKDIR /workspace