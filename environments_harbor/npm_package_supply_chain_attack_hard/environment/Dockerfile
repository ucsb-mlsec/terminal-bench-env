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
    nodejs \
    npm \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/suspicious-package/scripts \
    /workspace/suspicious-package/lib \
    /workspace/suspicious-package/package-versions/v2.3.4 \
    /workspace/suspicious-package/test

COPY suspicious-package/index.js /workspace/suspicious-package/index.js
COPY suspicious-package/scripts/install.js /workspace/suspicious-package/scripts/install.js
COPY suspicious-package/lib/validator.js /workspace/suspicious-package/lib/validator.js
COPY suspicious-package/lib/crypto-utils.js /workspace/suspicious-package/lib/crypto-utils.js
COPY suspicious-package/lib/parser.js /workspace/suspicious-package/lib/parser.js
COPY suspicious-package/package-versions/v2.3.4/index.js /workspace/suspicious-package/package-versions/v2.3.4/index.js
COPY suspicious-package/README.md /workspace/suspicious-package/README.md
COPY suspicious-package/.npmignore /workspace/suspicious-package/.npmignore
COPY suspicious-package/test/validator.test.js /workspace/suspicious-package/test/validator.test.js

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
