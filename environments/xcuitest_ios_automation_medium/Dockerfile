FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/UITests /workspace/config /workspace/data

COPY config/test_config.json /workspace/test_config.json
COPY data/app_ui_elements.json /workspace/app_ui_elements.json
COPY UITests/LoginTests.swift /workspace/UITests/LoginTests.swift
COPY UITests/HomeTests.swift /workspace/UITests/HomeTests.swift
COPY UITests/ProfileTests.swift /workspace/UITests/ProfileTests.swift
COPY UITests/SettingsTests.swift /workspace/UITests/SettingsTests.swift

WORKDIR /workspace