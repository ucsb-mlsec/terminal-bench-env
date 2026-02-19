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
    apt-get install -y tini && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/app

COPY requirements.txt /workspace/requirements.txt
COPY app/processor.py /workspace/app/processor.py
COPY app/worker.sh /workspace/app/worker.sh

RUN pip3 install --no-cache-dir --break-system-packages -r /workspace/requirements.txt

RUN chmod +x /workspace/app/worker.sh

WORKDIR /workspace