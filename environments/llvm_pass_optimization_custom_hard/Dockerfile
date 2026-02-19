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
    llvm \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/ir_samples

COPY data/ir_samples/basic_constant.ll /workspace/ir_samples/basic_constant.ll
COPY data/ir_samples/compute_kernel.ll /workspace/ir_samples/compute_kernel.ll
COPY data/ir_samples/loop_invariant.ll /workspace/ir_samples/loop_invariant.ll
COPY data/ir_samples/matrix_multiply.ll /workspace/ir_samples/matrix_multiply.ll
COPY data/ir_samples/pure_constant.ll /workspace/ir_samples/pure_constant.ll
COPY data/ir_samples/string_process.ll /workspace/ir_samples/string_process.ll
COPY data/ir_samples/dynamic_array.ll /workspace/ir_samples/dynamic_array.ll
COPY data/ir_samples/conditional_branch.ll /workspace/ir_samples/conditional_branch.ll
COPY data/ir_samples/simple_arithmetic.ll /workspace/ir_samples/simple_arithmetic.ll

WORKDIR /workspace