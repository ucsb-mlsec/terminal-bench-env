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
    ruby \
    ruby-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN gem install fluentd --no-document

RUN mkdir -p /var/fluentd_incident/app_logs /home

COPY config/fluent.conf /var/fluentd_incident/fluent.conf
COPY data/error.log /var/fluentd_incident/error.log
COPY data/metrics.log /var/fluentd_incident/metrics.log
COPY data/app_logs/application.log /var/fluentd_incident/app_logs/application.log

WORKDIR /var/fluentd_incident