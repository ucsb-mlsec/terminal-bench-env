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
    php-cli \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y php curl && \
    pip3 install --no-cache-dir --break-system-packages requests && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /var/www/html /home/ctf /tmp

COPY var/www/html/index.php /var/www/html/index.php
COPY home/ctf/flag.txt /home/ctf/flag.txt

RUN chmod 644 /var/www/html/index.php /home/ctf/flag.txt && \
    touch /tmp/app.log && \
    chmod 666 /tmp/app.log

WORKDIR /workspace