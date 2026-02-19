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
    php \
    php-cli \
    php-mbstring \
    php-xml \
    php-xdebug \
    composer && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y php php-xdebug && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/calculator-service/src \
    /workspace/calculator-service/tests \
    /workspace/calculator-service/vendor

COPY workspace/calculator-service/src/Calculator.php /workspace/calculator-service/src/Calculator.php
COPY workspace/calculator-service/tests/CalculatorTest.php /workspace/calculator-service/tests/CalculatorTest.php
COPY workspace/calculator-service/phpunit.xml /workspace/calculator-service/phpunit.xml
COPY workspace/calculator-service/composer.json /workspace/calculator-service/composer.json

WORKDIR /workspace/calculator-service