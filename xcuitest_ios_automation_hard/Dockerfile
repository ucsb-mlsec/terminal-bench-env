FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/ios_tests

COPY ios_tests/LoginTests.swift /workspace/ios_tests/LoginTests.swift
COPY ios_tests/HomeScreenTests.swift /workspace/ios_tests/HomeScreenTests.swift
COPY ios_tests/SearchTests.swift /workspace/ios_tests/SearchTests.swift
COPY ios_tests/SettingsTests.swift /workspace/ios_tests/SettingsTests.swift
COPY ios_tests/ProfileTests.swift /workspace/ios_tests/ProfileTests.swift

WORKDIR /workspace