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
    g++ \
    libboost-python-dev \
    libboost-all-dev \
    python3-dev \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y libboost-python-dev g++ python3-dev && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/matrix_ops

COPY matrix_extension.cpp /workspace/matrix_ops/matrix_extension.cpp
COPY setup.py /workspace/matrix_ops/setup.py

RUN cat > /workspace/matrix_ops/test_module.py << 'EOF'
#!/usr/bin/env python3
import sys

try:
    import matrix_ops
    print("Module imported successfully")
    print("ALL TESTS PASSED")
except Exception as e:
    print(f"Failed to import module: {e}")
    sys.exit(1)
EOF

RUN cat > /workspace/matrix_ops/README.md << 'EOF'
# Matrix Operations C++ Extension

A Python extension module for matrix operations using Boost.Python.

## Building
Run: python3 setup.py build_ext --inplace

## Testing
Run: python3 test_module.py
EOF

WORKDIR /workspace/matrix_ops