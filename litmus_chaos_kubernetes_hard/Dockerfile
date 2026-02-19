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
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages pyyaml

RUN mkdir -p /workspace/chaos-experiments /tmp

COPY chaos-experiments/experiment-01.yaml /workspace/chaos-experiments/experiment-01.yaml
COPY chaos-experiments/experiment-02.yaml /workspace/chaos-experiments/experiment-02.yaml
COPY chaos-experiments/experiment-03.yaml /workspace/chaos-experiments/experiment-03.yaml
COPY chaos-experiments/experiment-04.yaml /workspace/chaos-experiments/experiment-04.yaml
COPY chaos-experiments/experiment-05.yaml /workspace/chaos-experiments/experiment-05.yaml
COPY chaos-experiments/experiment-06.yaml /workspace/chaos-experiments/experiment-06.yaml
COPY chaos-experiments/experiment-07.yaml /workspace/chaos-experiments/experiment-07.yaml
COPY chaos-experiments/experiment-08.yaml /workspace/chaos-experiments/experiment-08.yaml
COPY chaos-experiments/experiment-09.yaml /workspace/chaos-experiments/experiment-09.yaml
COPY chaos-experiments/experiment-10.yaml /workspace/chaos-experiments/experiment-10.yaml

WORKDIR /workspace