FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/problems

COPY data/problems/problem_01.txt /workspace/problems/problem_01.txt
COPY data/problems/problem_02.txt /workspace/problems/problem_02.txt
COPY data/problems/problem_03.txt /workspace/problems/problem_03.txt
COPY data/problems/problem_04.txt /workspace/problems/problem_04.txt
COPY data/problems/problem_05.txt /workspace/problems/problem_05.txt
COPY data/problems/problem_06.txt /workspace/problems/problem_06.txt
COPY data/problems/problem_07.txt /workspace/problems/problem_07.txt
COPY data/problems/problem_08.txt /workspace/problems/problem_08.txt
COPY data/problems/problem_09.txt /workspace/problems/problem_09.txt
COPY data/problems/problem_10.txt /workspace/problems/problem_10.txt
COPY data/problems/problem_11.txt /workspace/problems/problem_11.txt
COPY data/problems/problem_12.txt /workspace/problems/problem_12.txt

WORKDIR /workspace