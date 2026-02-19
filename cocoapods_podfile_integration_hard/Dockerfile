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
    apt-get install -y ruby ruby-dev build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

RUN gem install cocoapods

RUN mkdir -p /workspace/ios_project/WeatherApp /workspace/ios_project/WeatherAppTests /workspace/solution

COPY ios_project/Podfile /workspace/ios_project/Podfile
COPY ios_project/WeatherApp/ViewController.swift /workspace/ios_project/WeatherApp/ViewController.swift
COPY ios_project/WeatherApp/NetworkManager.swift /workspace/ios_project/WeatherApp/NetworkManager.swift
COPY ios_project/WeatherAppTests/WeatherAppTests.swift /workspace/ios_project/WeatherAppTests/WeatherAppTests.swift

WORKDIR /workspace