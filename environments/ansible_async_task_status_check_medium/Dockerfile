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
    apt-get install -y jq && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y findutils && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /.ansible_async \
    /var/tmp/.ansible_async \
    /home/ansible/.ansible_async \
    /root/.ansible_async \
    /tmp/.ansible_async

COPY data/.ansible_async/123456789.12345 /.ansible_async/123456789.12345
COPY data/.ansible_async/987654321.54321 /.ansible_async/987654321.54321
COPY data/.ansible_async/555555555.55555 /.ansible_async/555555555.55555
COPY data/.ansible_async/111222333.44444 /.ansible_async/111222333.44444
COPY data/.ansible_async/222333444.55566 /.ansible_async/222333444.55566
COPY data/.ansible_async/444555666.77788 /.ansible_async/444555666.77788
COPY data/var/tmp/.ansible_async/666777888.99999 /var/tmp/.ansible_async/666777888.99999
COPY data/home/ansible/.ansible_async/333444555.66677 /home/ansible/.ansible_async/333444555.66677
COPY data/root/.ansible_async/777888999.11122 /root/.ansible_async/777888999.11122
COPY data/tmp/.ansible_async/888999000.22233 /tmp/.ansible_async/888999000.22233

WORKDIR /workspace