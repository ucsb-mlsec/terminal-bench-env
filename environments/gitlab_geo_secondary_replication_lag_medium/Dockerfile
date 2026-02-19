FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /var/gitlab-diagnostics /root

COPY data/gitlab-app.log /var/gitlab-diagnostics/gitlab-app.log
COPY data/gitaly.log /var/gitlab-diagnostics/gitaly.log
COPY data/sidekiq.log /var/gitlab-diagnostics/sidekiq.log
COPY data/system-resources.txt /var/gitlab-diagnostics/system-resources.txt
COPY data/gitlab.yml /var/gitlab-diagnostics/gitlab.yml
COPY data/gitaly-config.toml /var/gitlab-diagnostics/gitaly-config.toml
COPY data/service-status.txt /var/gitlab-diagnostics/service-status.txt

WORKDIR /root