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
    wget \
    gnupg2 \
    apt-transport-https \
    openjdk-11-jdk && \
    wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | apt-key add - && \
    echo "deb https://artifacts.elastic.co/packages/8.x/apt stable main" | tee -a /etc/apt/sources.list.d/elastic-8.x.list && \
    apt-get update && \
    apt-get install -y logstash && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/logstash/conf.d /var/log/logstash /var/log /solution

COPY config/pipeline.conf /etc/logstash/conf.d/pipeline.conf
COPY data/sample_access.log /var/log/sample_access.log

RUN echo '{"filter_processing_time_ms": 250, "worker_utilization": 45, "batch_efficiency": 30, "events_per_second": 500, "gc_pauses_per_minute": 15, "memory_usage_percent": 85, "bottlenecks": ["grok_parsing_slow", "low_batch_size", "insufficient_workers", "inefficient_mutate_operations"]}' > /var/log/logstash/performance_metrics.json

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
