FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install Ruby and dependencies
RUN apt-get update && \
    apt-get install -y ruby ruby-dev build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install FactoryBot gem
RUN gem install factory_bot --no-document

# Create necessary directories
RUN mkdir -p /workspace/models /workspace/solution

# Copy model files
COPY models/weapon.rb /workspace/models/weapon.rb
COPY models/armor.rb /workspace/models/armor.rb
COPY models/character.rb /workspace/models/character.rb

# Set working directory
WORKDIR /workspace