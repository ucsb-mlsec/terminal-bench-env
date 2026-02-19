FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/test_files/dir1 /workspace/test_files/dir2

COPY test_files/dir1/file1.txt /workspace/test_files/dir1/file1.txt
COPY test_files/dir1/file2.txt /workspace/test_files/dir1/file2.txt
COPY test_files/dir2/file2_link.txt /workspace/test_files/dir2/file2_link.txt
COPY test_files/dir2/file3.txt /workspace/test_files/dir2/file3.txt
COPY test_files/dir1/file4.txt /workspace/test_files/dir1/file4.txt
COPY test_files/dir1/file4_another.txt /workspace/test_files/dir1/file4_another.txt

WORKDIR /workspace