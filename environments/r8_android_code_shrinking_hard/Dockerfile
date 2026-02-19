FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/r8-analysis/source_snippets /workspace/solution

COPY data/crash_logs.txt /workspace/r8-analysis/crash_logs.txt
COPY data/library_manifest.txt /workspace/r8-analysis/library_manifest.txt
COPY source_snippets/ReflectionUsage.java /workspace/r8-analysis/source_snippets/ReflectionUsage.java
COPY source_snippets/SerializableModels.java /workspace/r8-analysis/source_snippets/SerializableModels.java
COPY source_snippets/NativeBindings.java /workspace/r8-analysis/source_snippets/NativeBindings.java
COPY source_snippets/LibraryCallbacks.java /workspace/r8-analysis/source_snippets/LibraryCallbacks.java
COPY source_snippets/AnnotatedClasses.java /workspace/r8-analysis/source_snippets/AnnotatedClasses.java

WORKDIR /workspace