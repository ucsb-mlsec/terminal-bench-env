FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install python3-pip and create python symlink once
RUN apt-get update && \
    apt-get install -y python3-pip git && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1 pyyaml

RUN mkdir -p /.github/workflows /tmp

# Create the corrected workflow file
RUN cat > /.github/workflows/ci.yml << 'EOF'
name: CI

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ['3.8', '3.9', '3.10', '3.11']
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies (Unix)
      if: runner.os != 'Windows'
      run: |
        python -m pip install --upgrade pip
        pip install pytest
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
    
    - name: Install dependencies (Windows)
      if: runner.os == 'Windows'
      run: |
        python -m pip install --upgrade pip
        pip install pytest
        if (Test-Path requirements.txt) { pip install -r requirements.txt }
      shell: pwsh
    
    - name: Run tests
      run: pytest
EOF

# Copy the fixed workflow to /tmp as required
RUN cp /.github/workflows/ci.yml /tmp/fixed_ci.yml

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
