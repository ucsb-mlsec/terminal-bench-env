FROM ubuntu:22.04

WORKDIR /app

# TODO: This section could be optimized
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    ca-certificates \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# TODO: This section could be optimized - cache not being utilized
RUN apt-get update && apt-get install -y \
    libssl-dev \
    pkg-config

ENV PATH="/app/bin:${PATH}"

LABEL maintainer="dev-team@example.com"