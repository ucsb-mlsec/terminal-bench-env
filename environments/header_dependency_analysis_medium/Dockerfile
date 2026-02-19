FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/cpp_project/core \
    /workspace/cpp_project/utils \
    /workspace/cpp_project/models \
    /workspace/cpp_project/network \
    /workspace/cpp_project/api

COPY cpp_project/core/database.h /workspace/cpp_project/core/database.h
COPY cpp_project/core/connection.h /workspace/cpp_project/core/connection.h
COPY cpp_project/core/logger.h /workspace/cpp_project/core/logger.h
COPY cpp_project/core/config.h /workspace/cpp_project/core/config.h
COPY cpp_project/utils/formatter.h /workspace/cpp_project/utils/formatter.h
COPY cpp_project/utils/parser.h /workspace/cpp_project/utils/parser.h
COPY cpp_project/utils/validator.h /workspace/cpp_project/utils/validator.h
COPY cpp_project/models/user.h /workspace/cpp_project/models/user.h
COPY cpp_project/models/account.h /workspace/cpp_project/models/account.h
COPY cpp_project/network/client.hpp /workspace/cpp_project/network/client.hpp
COPY cpp_project/network/server.hpp /workspace/cpp_project/network/server.hpp
COPY cpp_project/api/handler.h /workspace/cpp_project/api/handler.h
COPY cpp_project/main.cpp /workspace/cpp_project/main.cpp

WORKDIR /workspace