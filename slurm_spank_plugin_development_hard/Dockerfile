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
    apt-get install -y binutils file && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/plugins /solution

COPY plugins/valid_plugin1.so /workspace/plugins/valid_plugin1.so
COPY plugins/valid_plugin2.so /workspace/plugins/valid_plugin2.so
COPY plugins/missing_plugin_name.so /workspace/plugins/missing_plugin_name.so
COPY plugins/missing_plugin_type.so /workspace/plugins/missing_plugin_type.so
COPY plugins/missing_plugin_version.so /workspace/plugins/missing_plugin_version.so
COPY plugins/wrong_plugin_type.so /workspace/plugins/wrong_plugin_type.so
COPY plugins/valid_plugin3.so /workspace/plugins/valid_plugin3.so
COPY plugins/missing_all_symbols.so /workspace/plugins/missing_all_symbols.so

WORKDIR /workspace