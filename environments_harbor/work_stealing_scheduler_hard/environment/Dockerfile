FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/task_processor/workloads

COPY scheduler.py /opt/task_processor/scheduler.py
COPY simulator.py /opt/task_processor/simulator.py
COPY run_validation.py /opt/task_processor/run_validation.py
COPY data/workload_uniform.json /opt/task_processor/workloads/workload_uniform.json
COPY data/workload_skewed.json /opt/task_processor/workloads/workload_skewed.json
COPY data/workload_extreme.json /opt/task_processor/workloads/workload_extreme.json
COPY data/workload_small.json /opt/task_processor/workloads/workload_small.json
COPY data/workload_variable.json /opt/task_processor/workloads/workload_variable.json

WORKDIR /opt/task_processor

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
