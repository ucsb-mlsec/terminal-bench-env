FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/exam_data /workspace/solution

COPY data/exam_data/courses.txt /workspace/exam_data/courses.txt
COPY data/exam_data/conflicts.txt /workspace/exam_data/conflicts.txt
COPY data/exam_data/constraints.txt /workspace/exam_data/constraints.txt

RUN pip3 install --no-cache-dir --break-system-packages ortools

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
