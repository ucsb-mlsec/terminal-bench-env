FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages pyyaml

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/streaming_app/config \
    /opt/streaming_app/scripts \
    /opt/streaming_app/deploy \
    /opt/streaming_app/src

COPY config/streaming_app.properties /opt/streaming_app/config/streaming_app.properties
COPY config/application.yaml /opt/streaming_app/config/application.yaml
COPY config/streams.json /opt/streaming_app/config/streams.json
COPY config/environment.conf /opt/streaming_app/config/environment.conf
COPY config/pipeline_settings.ini /opt/streaming_app/config/pipeline_settings.ini
COPY scripts/start_streaming.sh /opt/streaming_app/scripts/start_streaming.sh
COPY scripts/restart_job.sh /opt/streaming_app/scripts/restart_job.sh
COPY deploy/deployment.yaml /opt/streaming_app/deploy/deployment.yaml
COPY src/config_loader.py /opt/streaming_app/src/config_loader.py

RUN chmod +x /opt/streaming_app/scripts/*.sh

WORKDIR /opt/streaming_app