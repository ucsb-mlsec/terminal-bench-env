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
    nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y nvidia-cuda-toolkit && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/cuda_snippets

COPY cuda_snippets/snippet_01.cu /workspace/cuda_snippets/snippet_01.cu
COPY cuda_snippets/snippet_02.cu /workspace/cuda_snippets/snippet_02.cu
COPY cuda_snippets/snippet_03.cu /workspace/cuda_snippets/snippet_03.cu
COPY cuda_snippets/snippet_04.cu /workspace/cuda_snippets/snippet_04.cu
COPY cuda_snippets/snippet_05.cu /workspace/cuda_snippets/snippet_05.cu
COPY cuda_snippets/snippet_06.cu /workspace/cuda_snippets/snippet_06.cu
COPY cuda_snippets/snippet_07.cu /workspace/cuda_snippets/snippet_07.cu
COPY cuda_snippets/snippet_08.cu /workspace/cuda_snippets/snippet_08.cu
COPY cuda_snippets/snippet_09.cu /workspace/cuda_snippets/snippet_09.cu
COPY cuda_snippets/snippet_10.cu /workspace/cuda_snippets/snippet_10.cu
COPY cuda_snippets/snippet_11.cu /workspace/cuda_snippets/snippet_11.cu
COPY cuda_snippets/snippet_12.cu /workspace/cuda_snippets/snippet_12.cu

WORKDIR /workspace