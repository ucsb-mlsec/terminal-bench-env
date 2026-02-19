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
    libxml2-utils \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/ios_app/MyApp.xcodeproj \
    /workspace/ios_app/MyApp

COPY ios_app/MyApp.xcodeproj/project.pbxproj /workspace/ios_app/MyApp.xcodeproj/project.pbxproj
COPY ios_app/MyApp/Info.plist /workspace/ios_app/MyApp/Info.plist
COPY ios_app/MyApp/AppDelegate.swift /workspace/ios_app/MyApp/AppDelegate.swift
COPY ios_app/MyApp/ViewController.swift /workspace/ios_app/MyApp/ViewController.swift
COPY ios_app/MyApp.entitlements /workspace/ios_app/MyApp.entitlements
COPY ios_app/ExportOptions.plist /workspace/ios_app/ExportOptions.plist

WORKDIR /workspace