FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/math_ops/dist \
    /workspace/math_ops/build/temp.linux-x86_64-3.9 \
    /workspace/math_ops/target/wheels \
    /workspace/math_ops/src

COPY math_ops/setup.py /workspace/math_ops/setup.py
COPY math_ops/Cargo.toml /workspace/math_ops/Cargo.toml
COPY math_ops/src/lib.rs /workspace/math_ops/src/lib.rs
COPY math_ops/pyproject.toml /workspace/math_ops/pyproject.toml
COPY math_ops/__init__.py /workspace/math_ops/__init__.py
COPY math_ops/dist/math_ops-1.2.3-cp39-cp39-linux_x86_64.whl /workspace/math_ops/dist/math_ops-1.2.3-cp39-cp39-linux_x86_64.whl
COPY math_ops/build/temp.linux-x86_64-3.9/debug.log /workspace/math_ops/build/temp.linux-x86_64-3.9/debug.log
COPY math_ops/target/wheels/build_info.txt /workspace/math_ops/target/wheels/build_info.txt

WORKDIR /workspace