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

RUN pip3 install --no-cache-dir --break-system-packages numpy pandas scipy

RUN mkdir -p /workspace/coverage_data /workspace

COPY data/coverage_data/SampleA_treatment.txt /workspace/coverage_data/SampleA_treatment.txt
COPY data/coverage_data/SampleA_control.txt /workspace/coverage_data/SampleA_control.txt
COPY data/coverage_data/SampleB_treatment.txt /workspace/coverage_data/SampleB_treatment.txt
COPY data/coverage_data/SampleB_control.txt /workspace/coverage_data/SampleB_control.txt
COPY data/coverage_data/SampleC_treatment.txt /workspace/coverage_data/SampleC_treatment.txt
COPY data/coverage_data/SampleC_control.txt /workspace/coverage_data/SampleC_control.txt

WORKDIR /workspace