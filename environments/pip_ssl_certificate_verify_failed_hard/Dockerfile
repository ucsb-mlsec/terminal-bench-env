FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /etc/ssl/certs \
    && mkdir -p /opt/project_env/bin \
    && mkdir -p /tmp

COPY etc/ssl/certs/corporate-ca.crt /etc/ssl/certs/corporate-ca.crt
COPY opt/project_env/pyvenv.cfg /opt/project_env/pyvenv.cfg
COPY opt/project_env/bin/activate /opt/project_env/bin/activate
COPY opt/project_env/bin/pip /opt/project_env/bin/pip
COPY opt/project_env/bin/python /opt/project_env/bin/python

RUN chmod +r /etc/ssl/certs/corporate-ca.crt \
    && chmod +x /opt/project_env/bin/pip \
    && chmod +x /opt/project_env/bin/python \
    && chmod +r /opt/project_env/bin/activate

WORKDIR /workspace