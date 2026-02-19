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
    apt-get install -y \
    gcc \
    make \
    gdb \
    file \
    binutils && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc make libc6-dev && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/hooking_lib/src && \
    mkdir -p /var/crash && \
    mkdir -p /var/log && \
    mkdir -p /root

COPY opt/hooking_lib/src/hooks.c /opt/hooking_lib/src/hooks.c
COPY opt/hooking_lib/src/hook_loader.c /opt/hooking_lib/src/hook_loader.c
COPY opt/hooking_lib/Makefile /opt/hooking_lib/Makefile
COPY opt/hooking_lib/libhook.so /opt/hooking_lib/libhook.so
COPY opt/hooking_lib/test_program.c /opt/hooking_lib/test_program.c
COPY opt/hooking_lib/README.txt /opt/hooking_lib/README.txt
COPY opt/hooking_lib/config.txt /opt/hooking_lib/config.txt
COPY var/crash/core.12345 /var/crash/core.12345
COPY var/crash/core.12346 /var/crash/core.12346
COPY var/log/syslog /var/log/syslog
COPY var/log/audit.log /var/log/audit.log

WORKDIR /opt/hooking_lib