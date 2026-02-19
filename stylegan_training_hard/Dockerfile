FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/stylegan_logs

COPY data/stylegan_logs/run_1.log /workspace/stylegan_logs/run_1.log
COPY data/stylegan_logs/run_2.log /workspace/stylegan_logs/run_2.log
COPY data/stylegan_logs/run_3.log /workspace/stylegan_logs/run_3.log
COPY data/stylegan_logs/run_4.log /workspace/stylegan_logs/run_4.log
COPY data/stylegan_logs/run_5.log /workspace/stylegan_logs/run_5.log
COPY data/stylegan_logs/run_6.log /workspace/stylegan_logs/run_6.log
COPY data/stylegan_logs/run_7.log /workspace/stylegan_logs/run_7.log
COPY data/stylegan_logs/run_8.log /workspace/stylegan_logs/run_8.log
COPY data/stylegan_logs/run_9.log /workspace/stylegan_logs/run_9.log
COPY data/stylegan_logs/run_10.log /workspace/stylegan_logs/run_10.log
COPY data/stylegan_logs/run_11.log /workspace/stylegan_logs/run_11.log
COPY data/stylegan_logs/run_12.log /workspace/stylegan_logs/run_12.log
COPY data/stylegan_logs/run_13.log /workspace/stylegan_logs/run_13.log
COPY data/stylegan_logs/run_14.log /workspace/stylegan_logs/run_14.log
COPY data/stylegan_logs/run_15.log /workspace/stylegan_logs/run_15.log

WORKDIR /workspace