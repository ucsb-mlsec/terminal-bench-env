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
    apt-get install -y cmake && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages pytest

RUN mkdir -p /workspace/datalib/tests /workspace/datalib/build

COPY CMakeLists.txt /workspace/datalib/CMakeLists.txt
COPY tests/__init__.py /workspace/datalib/tests/__init__.py
COPY tests/test_basic.py /workspace/datalib/tests/test_basic.py
COPY tests/test_data_processing.py /workspace/datalib/tests/test_data_processing.py

WORKDIR /workspace/datalib