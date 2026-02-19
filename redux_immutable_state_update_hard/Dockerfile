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
    apt-get install -y nodejs npm && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/cart-app/src/reducers /workspace/cart-app/tests

COPY src/reducers/cartReducer.js /workspace/cart-app/src/reducers/cartReducer.js
COPY tests/cartReducer.test.js /workspace/cart-app/tests/cartReducer.test.js
COPY package.json /workspace/cart-app/package.json

WORKDIR /workspace/cart-app

RUN npm install