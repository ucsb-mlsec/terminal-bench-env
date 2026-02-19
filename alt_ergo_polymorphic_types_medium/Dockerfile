FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/formulas

COPY data/formulas/formula_1.txt /workspace/formulas/formula_1.txt
COPY data/formulas/formula_2.txt /workspace/formulas/formula_2.txt
COPY data/formulas/formula_3.txt /workspace/formulas/formula_3.txt
COPY data/formulas/formula_4.txt /workspace/formulas/formula_4.txt
COPY data/formulas/formula_5.txt /workspace/formulas/formula_5.txt
COPY data/formulas/formula_6.txt /workspace/formulas/formula_6.txt
COPY data/formulas/formula_7.txt /workspace/formulas/formula_7.txt
COPY data/formulas/formula_8.txt /workspace/formulas/formula_8.txt

WORKDIR /workspace