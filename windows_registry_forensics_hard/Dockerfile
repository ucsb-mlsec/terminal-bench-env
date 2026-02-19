FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /forensics/registry_exports /solution

COPY data/forensics/registry_exports/HKLM_SOFTWARE.reg /forensics/registry_exports/HKLM_SOFTWARE.reg
COPY data/forensics/registry_exports/HKCU_SOFTWARE.reg /forensics/registry_exports/HKCU_SOFTWARE.reg
COPY data/forensics/registry_exports/SERVICES.reg /forensics/registry_exports/SERVICES.reg
COPY data/forensics/registry_exports/TASK_SCHEDULER.reg /forensics/registry_exports/TASK_SCHEDULER.reg
COPY data/forensics/registry_exports/SHELL_EXTENSIONS.reg /forensics/registry_exports/SHELL_EXTENSIONS.reg

WORKDIR /forensics