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
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages numpy scipy

RUN mkdir -p /workspace/fep_data

COPY data/fep_window_0.fepout /workspace/fep_data/fep_window_0.fepout
COPY data/fep_window_1.fepout /workspace/fep_data/fep_window_1.fepout
COPY data/fep_window_2.fepout /workspace/fep_data/fep_window_2.fepout
COPY data/fep_window_3.fepout /workspace/fep_data/fep_window_3.fepout
COPY data/fep_window_4.fepout /workspace/fep_data/fep_window_4.fepout
COPY data/fep_window_5.fepout /workspace/fep_data/fep_window_5.fepout
COPY data/fep_window_6.fepout /workspace/fep_data/fep_window_6.fepout
COPY data/fep_window_7.fepout /workspace/fep_data/fep_window_7.fepout
COPY data/fep_window_8.fepout /workspace/fep_data/fep_window_8.fepout
COPY data/fep_window_9.fepout /workspace/fep_data/fep_window_9.fepout
COPY data/fep_window_10.fepout /workspace/fep_data/fep_window_10.fepout
COPY data/fep_window_11.fepout /workspace/fep_data/fep_window_11.fepout
COPY data/fep_window_12.fepout /workspace/fep_data/fep_window_12.fepout
COPY data/fep_window_13.fepout /workspace/fep_data/fep_window_13.fepout
COPY data/fep_window_14.fepout /workspace/fep_data/fep_window_14.fepout
COPY data/fep_window_15.fepout /workspace/fep_data/fep_window_15.fepout
COPY data/fep_window_16.fepout /workspace/fep_data/fep_window_16.fepout
COPY data/fep_window_17.fepout /workspace/fep_data/fep_window_17.fepout
COPY data/fep_window_18.fepout /workspace/fep_data/fep_window_18.fepout
COPY data/fep_window_19.fepout /workspace/fep_data/fep_window_19.fepout

WORKDIR /workspace