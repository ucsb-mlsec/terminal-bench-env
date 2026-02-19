FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install OpenWrt build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    clang \
    flex \
    bison \
    g++ \
    gawk \
    gcc-multilib \
    g++-multilib \
    gettext \
    git \
    libncurses5-dev \
    libssl-dev \
    rsync \
    unzip \
    zlib1g-dev \
    file \
    wget \
    subversion \
    time \
    libelf-dev \
    python3-setuptools \
    patch \
    quilt \
    xsltproc \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y build-essential gawk unzip libncurses5-dev zlib1g-dev git wget python3 file && \
    rm -rf /var/lib/apt/lists/*

# Create directory structure
RUN mkdir -p /workspace/openwrt-build/package/custom-package/src \
    /workspace/openwrt-build/target/linux/ramips/mt7621 \
    /workspace/openwrt-build/include \
    /workspace/openwrt-build/scripts \
    /workspace/openwrt-build/package/kernel/linux/modules \
    /workspace/openwrt-build/target/linux/ramips/image

# Copy build files
COPY openwrt-build/.config /workspace/openwrt-build/.config
COPY openwrt-build/package/custom-package/Makefile /workspace/openwrt-build/package/custom-package/Makefile
COPY openwrt-build/target/linux/ramips/mt7621/config-default /workspace/openwrt-build/target/linux/ramips/mt7621/config-default
COPY openwrt-build/include/custom.mk /workspace/openwrt-build/include/custom.mk
COPY openwrt-build/scripts/build_wrapper.sh /workspace/openwrt-build/scripts/build_wrapper.sh
COPY openwrt-build/feeds.conf /workspace/openwrt-build/feeds.conf
COPY openwrt-build/package/custom-package/src/main.c /workspace/openwrt-build/package/custom-package/src/main.c
COPY openwrt-build/rules.mk /workspace/openwrt-build/rules.mk
COPY openwrt-build/package/kernel/linux/modules/netfilter.mk /workspace/openwrt-build/package/kernel/linux/modules/netfilter.mk
COPY openwrt-build/target/linux/ramips/image/mt7621.mk /workspace/openwrt-build/target/linux/ramips/image/mt7621.mk

RUN chmod +x /workspace/openwrt-build/scripts/build_wrapper.sh

WORKDIR /workspace/openwrt-build