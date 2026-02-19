FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /home/user/chunks

COPY data/chunks/part_0_5242879.bin /home/user/chunks/part_0_5242879.bin
COPY data/chunks/part_5242880_10485759.bin /home/user/chunks/part_5242880_10485759.bin
COPY data/chunks/part_10485760_15728639.bin /home/user/chunks/part_10485760_15728639.bin
COPY data/chunks/part_15728640_20971519.bin /home/user/chunks/part_15728640_20971519.bin
COPY data/chunks/part_20971520_26214399.bin /home/user/chunks/part_20971520_26214399.bin
COPY data/chunks/part_26214400_31457279.bin /home/user/chunks/part_26214400_31457279.bin
COPY data/chunks/part_31457280_36700159.bin /home/user/chunks/part_31457280_36700159.bin
COPY data/chunks/part_36700160_41943039.bin /home/user/chunks/part_36700160_41943039.bin
COPY data/chunks/part_41943040_47185919.bin /home/user/chunks/part_41943040_47185919.bin
COPY data/chunks/part_47185920_52428799.bin /home/user/chunks/part_47185920_52428799.bin
COPY data/chunks/expected_checksum.txt /home/user/chunks/expected_checksum.txt

WORKDIR /home/user