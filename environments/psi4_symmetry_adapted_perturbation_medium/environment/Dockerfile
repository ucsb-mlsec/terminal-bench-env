FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/sapt_outputs

COPY data/sapt_outputs/complex_A_3.5.json /workspace/sapt_outputs/complex_A_3.5.json
COPY data/sapt_outputs/complex_B_3.0.json /workspace/sapt_outputs/complex_B_3.0.json
COPY data/sapt_outputs/complex_B_4.0.json /workspace/sapt_outputs/complex_B_4.0.json
COPY data/sapt_outputs/complex_C_3.2.json /workspace/sapt_outputs/complex_C_3.2.json
COPY data/sapt_outputs/complex_C_3.8.json /workspace/sapt_outputs/complex_C_3.8.json
COPY data/sapt_outputs/complex_D_2.8.json /workspace/sapt_outputs/complex_D_2.8.json
COPY data/sapt_outputs/complex_A_4.5.json /workspace/sapt_outputs/complex_A_4.5.json

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
