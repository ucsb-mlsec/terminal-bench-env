FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/processor/batches /workspace/solution

COPY process.py /workspace/processor/process.py
COPY data/batch_metadata.json /workspace/processor/batch_metadata.json
COPY data/batches/batch_001.dat /workspace/processor/batches/batch_001.dat
COPY data/batches/batch_002.dat /workspace/processor/batches/batch_002.dat
COPY data/batches/batch_003.dat /workspace/processor/batches/batch_003.dat
COPY data/batches/batch_004.dat /workspace/processor/batches/batch_004.dat
COPY data/batches/batch_005.dat /workspace/processor/batches/batch_005.dat
COPY data/batches/batch_006.dat /workspace/processor/batches/batch_006.dat
COPY data/batches/batch_007.dat /workspace/processor/batches/batch_007.dat
COPY data/batches/batch_008.dat /workspace/processor/batches/batch_008.dat
COPY data/batches/batch_009.dat /workspace/processor/batches/batch_009.dat
COPY data/batches/batch_010.dat /workspace/processor/batches/batch_010.dat
COPY data/batches/batch_011.dat /workspace/processor/batches/batch_011.dat
COPY data/batches/batch_012.dat /workspace/processor/batches/batch_012.dat
COPY data/batches/batch_013.dat /workspace/processor/batches/batch_013.dat
COPY data/batches/batch_014.dat /workspace/processor/batches/batch_014.dat
COPY data/batches/batch_015.dat /workspace/processor/batches/batch_015.dat

WORKDIR /workspace