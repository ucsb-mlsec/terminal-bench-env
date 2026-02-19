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
    apt-get install -y python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages torch torchvision numpy

RUN mkdir -p /workspace/model_files /workspace/output

COPY model_files/baseline_model.py /workspace/model_files/baseline_model.py
COPY model_files/generate_model.py /workspace/model_files/generate_model.py
COPY data/sample_inputs.npy /workspace/model_files/sample_inputs.npy

WORKDIR /workspace