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
    apt-get install -y protobuf-compiler && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages protobuf

RUN mkdir -p /workspace/data

COPY data/message.proto /workspace/data/message.proto
COPY data/transaction.proto /workspace/data/transaction.proto
COPY app.py /workspace/app.py
COPY requirements.txt /workspace/requirements.txt
COPY data_processor.py /workspace/data_processor.py

WORKDIR /workspace