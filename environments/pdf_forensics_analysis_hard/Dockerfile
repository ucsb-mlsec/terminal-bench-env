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
    apt-get install -y --no-install-recommends \
    python3-pip \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages \
    pycryptodome

RUN mkdir -p /workspace/pdf_objects

COPY data/pdf_objects/stream_1.dat /workspace/pdf_objects/stream_1.dat
COPY data/pdf_objects/stream_2.dat /workspace/pdf_objects/stream_2.dat
COPY data/pdf_objects/stream_3.dat /workspace/pdf_objects/stream_3.dat
COPY data/pdf_objects/stream_4.dat /workspace/pdf_objects/stream_4.dat
COPY data/pdf_objects/stream_5.dat /workspace/pdf_objects/stream_5.dat
COPY data/pdf_objects/stream_6.dat /workspace/pdf_objects/stream_6.dat
COPY data/pdf_objects/stream_7.dat /workspace/pdf_objects/stream_7.dat
COPY data/pdf_objects/stream_8.dat /workspace/pdf_objects/stream_8.dat
COPY data/pdf_objects/metadata.txt /workspace/pdf_objects/metadata.txt

WORKDIR /workspace