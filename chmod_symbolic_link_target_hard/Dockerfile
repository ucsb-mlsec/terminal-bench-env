FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/filemanager/nested/deep /tmp

COPY opt/filemanager/config1.txt /opt/filemanager/config1.txt
COPY opt/filemanager/config2.txt /opt/filemanager/config2.txt
COPY opt/filemanager/nested/config3.txt /opt/filemanager/nested/config3.txt
COPY opt/filemanager/nested/deep/config4.txt /opt/filemanager/nested/deep/config4.txt
COPY opt/filemanager/target.conf /opt/filemanager/target.conf

RUN ln -s /opt/filemanager/config1.txt /opt/filemanager/link1.txt && \
    ln -s target.conf /opt/filemanager/link2.txt && \
    ln -s target.conf /opt/filemanager/chain_link1.txt && \
    ln -s chain_link1.txt /opt/filemanager/chain_link2.txt && \
    ln -s ../config2.txt /opt/filemanager/nested/nested_link.txt && \
    ln -s /opt/filemanager/nested/config3.txt /opt/filemanager/nested/deep_link.txt && \
    ln -s ../../target.conf /opt/filemanager/nested/deep/upward_link.txt && \
    ln -s /nonexistent/file.txt /opt/filemanager/broken_link.txt && \
    ln -s /absolute/path/missing.txt /opt/filemanager/absolute_broken.txt && \
    ln -s ../chain_link2.txt /opt/filemanager/nested/chain_start.txt && \
    ln -s circular_b.txt /opt/filemanager/circular_a.txt && \
    ln -s circular_a.txt /opt/filemanager/circular_b.txt

RUN chmod 644 /opt/filemanager/config1.txt && \
    chmod 600 /opt/filemanager/config2.txt && \
    chmod 755 /opt/filemanager/nested/config3.txt && \
    chmod 640 /opt/filemanager/nested/deep/config4.txt && \
    chmod 666 /opt/filemanager/target.conf

WORKDIR /opt/filemanager