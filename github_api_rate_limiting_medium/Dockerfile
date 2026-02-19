FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /tmp/github_responses

COPY data/github_responses/facebook_react.json /tmp/github_responses/facebook_react.json
COPY data/github_responses/microsoft_vscode.json /tmp/github_responses/microsoft_vscode.json
COPY data/github_responses/google_tensorflow.json /tmp/github_responses/google_tensorflow.json
COPY data/github_responses/nodejs_node.json /tmp/github_responses/nodejs_node.json
COPY data/github_responses/python_cpython.json /tmp/github_responses/python_cpython.json
COPY data/github_responses/angular_angular.json /tmp/github_responses/angular_angular.json
COPY data/github_responses/rails_rails.json /tmp/github_responses/rails_rails.json
COPY data/github_responses/apple_swift.json /tmp/github_responses/apple_swift.json

WORKDIR /workspace