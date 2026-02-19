FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/source_data/subdir/nested

COPY data/source_data/file1.txt /workspace/source_data/file1.txt
COPY data/source_data/file2.txt /workspace/source_data/file2.txt
COPY data/source_data/file3.txt /workspace/source_data/file3.txt
COPY data/source_data/subdir/doc1.txt /workspace/source_data/subdir/doc1.txt
COPY data/source_data/subdir/doc2.txt /workspace/source_data/subdir/doc2.txt
COPY data/source_data/doc3.txt /workspace/source_data/doc3.txt
COPY data/source_data/data.bin /workspace/source_data/data.bin
COPY data/source_data/subdir/nested/deep1.txt /workspace/source_data/subdir/nested/deep1.txt
COPY data/source_data/subdir/nested/deep2.txt /workspace/source_data/subdir/nested/deep2.txt
COPY data/source_data/subdir/nested/deep3.txt /workspace/source_data/subdir/nested/deep3.txt

RUN ln -f /workspace/source_data/file1.txt /workspace/source_data/file2.txt && \
    ln -f /workspace/source_data/subdir/doc1.txt /workspace/source_data/subdir/doc2.txt && \
    ln -f /workspace/source_data/subdir/doc1.txt /workspace/source_data/doc3.txt && \
    ln -f /workspace/source_data/subdir/nested/deep1.txt /workspace/source_data/subdir/nested/deep2.txt

WORKDIR /workspace