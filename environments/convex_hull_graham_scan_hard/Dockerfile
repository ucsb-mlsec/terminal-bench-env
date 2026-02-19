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
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages scipy

RUN mkdir -p /workspace/sheets

COPY data/reference.txt /workspace/reference.txt
COPY data/sheets/sheet_1.txt /workspace/sheets/sheet_1.txt
COPY data/sheets/sheet_2.txt /workspace/sheets/sheet_2.txt
COPY data/sheets/sheet_3.txt /workspace/sheets/sheet_3.txt
COPY data/sheets/sheet_4.txt /workspace/sheets/sheet_4.txt
COPY data/sheets/sheet_5.txt /workspace/sheets/sheet_5.txt
COPY data/sheets/sheet_10.txt /workspace/sheets/sheet_10.txt
COPY data/sheets/sheet_15.txt /workspace/sheets/sheet_15.txt

WORKDIR /workspace