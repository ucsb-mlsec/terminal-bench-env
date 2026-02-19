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
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages pandas

RUN mkdir -p /home/agent/attack_data /home/agent/solution

COPY data/boot_disassembly.txt /home/agent/attack_data/boot_disassembly.txt
COPY data/power_trace.csv /home/agent/attack_data/power_trace.csv
COPY data/execution_trace.log /home/agent/attack_data/execution_trace.log

WORKDIR /home/agent