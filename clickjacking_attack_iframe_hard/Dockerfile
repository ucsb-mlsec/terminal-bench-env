FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /var/www/attack-server/styles \
    /var/www/attack-server/js \
    /var/www/attack-server/pages \
    /var/www/attack-server/phishing \
    /var/www/attack-server/assets \
    /var/www/attack-server/variants \
    /var/www/attack-server/scripts \
    /var/www/attack-server/admin \
    /tmp

COPY data/attack-server/index.html /var/www/attack-server/index.html
COPY data/attack-server/styles/overlay.css /var/www/attack-server/styles/overlay.css
COPY data/attack-server/js/frame-controller.js /var/www/attack-server/js/frame-controller.js
COPY data/attack-server/js/obfuscated.js /var/www/attack-server/js/obfuscated.js
COPY data/attack-server/pages/landing.html /var/www/attack-server/pages/landing.html
COPY data/attack-server/phishing/capture.html /var/www/attack-server/phishing/capture.html
COPY data/attack-server/success.html /var/www/attack-server/success.html
COPY data/attack-server/assets/config.json /var/www/attack-server/assets/config.json
COPY data/attack-server/variants/mobile.html /var/www/attack-server/variants/mobile.html
COPY data/attack-server/variants/social-media.html /var/www/attack-server/variants/social-media.html
COPY data/attack-server/scripts/redirect.js /var/www/attack-server/scripts/redirect.js
COPY data/attack-server/admin/credentials.txt /var/www/attack-server/admin/credentials.txt

WORKDIR /var/www/attack-server