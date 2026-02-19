FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/dataset/class_0 \
    /workspace/dataset/class_1 \
    /workspace/dataset/class_2 \
    /workspace/dataset/class_3 \
    /workspace/dataset/class_4 \
    /workspace/dataset/class_5

RUN pip3 install --no-cache-dir --break-system-packages numpy

COPY data/dataset/class_0/support.csv /workspace/dataset/class_0/support.csv
COPY data/dataset/class_0/query.csv /workspace/dataset/class_0/query.csv
COPY data/dataset/class_1/support.csv /workspace/dataset/class_1/support.csv
COPY data/dataset/class_1/query.csv /workspace/dataset/class_1/query.csv
COPY data/dataset/class_2/support.csv /workspace/dataset/class_2/support.csv
COPY data/dataset/class_2/query.csv /workspace/dataset/class_2/query.csv
COPY data/dataset/class_3/support.csv /workspace/dataset/class_3/support.csv
COPY data/dataset/class_3/query.csv /workspace/dataset/class_3/query.csv
COPY data/dataset/class_4/support.csv /workspace/dataset/class_4/support.csv
COPY data/dataset/class_4/query.csv /workspace/dataset/class_4/query.csv
COPY data/dataset/class_5/support.csv /workspace/dataset/class_5/support.csv
COPY data/dataset/class_5/query.csv /workspace/dataset/class_5/query.csv

WORKDIR /workspace