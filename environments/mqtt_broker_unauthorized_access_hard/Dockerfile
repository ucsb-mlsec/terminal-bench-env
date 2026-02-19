FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Create necessary directories
RUN mkdir -p /var/log/mqtt_incident /tmp

# Copy all generated files to the expected locations
COPY data/mqtt_messages.log /var/log/mqtt_incident/mqtt_messages.log
COPY data/client_connections.log /var/log/mqtt_incident/client_connections.log
COPY data/topic_subscriptions.log /var/log/mqtt_incident/topic_subscriptions.log
COPY data/mqtt_broker.conf /var/log/mqtt_incident/mqtt_broker.conf

# Set working directory
WORKDIR /var/log/mqtt_incident