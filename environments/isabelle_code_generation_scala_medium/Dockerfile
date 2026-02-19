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
    scala \
    openjdk-11-jdk-headless && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y scala && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/isabelle_output/subdir

COPY isabelle_output/QuickSort.scala /workspace/isabelle_output/QuickSort.scala
COPY isabelle_output/MergeSort.scala /workspace/isabelle_output/MergeSort.scala
COPY isabelle_output/InsertionSort.scala /workspace/isabelle_output/InsertionSort.scala
COPY isabelle_output/SortingProofs.scala /workspace/isabelle_output/SortingProofs.scala
COPY isabelle_output/generated_code.log /workspace/isabelle_output/generated_code.log
COPY isabelle_output/temp_build.txt /workspace/isabelle_output/temp_build.txt
COPY isabelle_output/subdir/ListUtils.scala /workspace/isabelle_output/subdir/ListUtils.scala

WORKDIR /workspace