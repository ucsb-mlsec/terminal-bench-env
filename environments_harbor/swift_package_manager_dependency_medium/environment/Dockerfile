FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/Sources /workspace/output/swift_package_manager_dependency_medium/files/Sources

COPY Package.swift /workspace/Package.swift
COPY Sources/main.swift /workspace/Sources/main.swift
COPY Sources/NetworkManager.swift /workspace/Sources/NetworkManager.swift
COPY Sources/DataParser.swift /workspace/Sources/DataParser.swift
COPY Package.swift /workspace/output/swift_package_manager_dependency_medium/files/Package.swift
COPY Sources/main.swift /workspace/output/swift_package_manager_dependency_medium/files/Sources/main.swift
COPY Sources/NetworkManager.swift /workspace/output/swift_package_manager_dependency_medium/files/Sources/NetworkManager.swift
COPY Sources/DataParser.swift /workspace/output/swift_package_manager_dependency_medium/files/Sources/DataParser.swift

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
