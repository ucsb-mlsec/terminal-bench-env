FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/textdata

COPY data/textdata/file1.txt /workspace/textdata/file1.txt
COPY data/textdata/file2.txt /workspace/textdata/file2.txt
COPY data/textdata/file3.txt /workspace/textdata/file3.txt
COPY data/textdata/file4.txt /workspace/textdata/file4.txt
COPY data/textdata/file5.txt /workspace/textdata/file5.txt
COPY data/textdata/file6.txt /workspace/textdata/file6.txt
COPY data/textdata/file7.txt /workspace/textdata/file7.txt
COPY data/textdata/file8.txt /workspace/textdata/file8.txt
COPY data/search_terms.txt /workspace/search_terms.txt

WORKDIR /workspace