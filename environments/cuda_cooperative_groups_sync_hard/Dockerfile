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
    apt-get install -y build-essential gcc make && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/parallel_compute/input_data && \
    mkdir -p /workspace/parallel_compute/reference_output && \
    mkdir -p /workspace/parallel_compute/test_results

COPY parallel_compute/refine.c /workspace/parallel_compute/refine.c
COPY parallel_compute/Makefile /workspace/parallel_compute/Makefile
COPY parallel_compute/data/dataset1.txt /workspace/parallel_compute/input_data/dataset1.txt
COPY parallel_compute/data/dataset2.txt /workspace/parallel_compute/input_data/dataset2.txt
COPY parallel_compute/data/dataset3.txt /workspace/parallel_compute/input_data/dataset3.txt
COPY parallel_compute/data/dataset1_expected.txt /workspace/parallel_compute/reference_output/dataset1_expected.txt
COPY parallel_compute/data/dataset2_expected.txt /workspace/parallel_compute/reference_output/dataset2_expected.txt
COPY parallel_compute/data/dataset3_expected.txt /workspace/parallel_compute/reference_output/dataset3_expected.txt
COPY parallel_compute/data/run1.txt /workspace/parallel_compute/test_results/run1.txt
COPY parallel_compute/data/run2.txt /workspace/parallel_compute/test_results/run2.txt
COPY parallel_compute/data/run3.txt /workspace/parallel_compute/test_results/run3.txt
COPY parallel_compute/data/run4.txt /workspace/parallel_compute/test_results/run4.txt
COPY parallel_compute/data/run5.txt /workspace/parallel_compute/test_results/run5.txt

WORKDIR /workspace