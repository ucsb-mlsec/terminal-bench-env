FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install Go and necessary tools
RUN apt-get update && \
    apt-get install -y golang-go git && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y golang && \
    rm -rf /var/lib/apt/lists/*

# Set Go environment variables
ENV GOPATH=/go
ENV PATH=$PATH:/usr/local/go/bin:$GOPATH/bin

# Install gomock/mockgen
RUN go install github.com/golang/mock/mockgen@v1.6.0

# Create necessary directories
RUN mkdir -p /home/developer/auth-service/internal/auth

# Copy project files
COPY home/developer/auth-service/go.mod /home/developer/auth-service/go.mod
COPY home/developer/auth-service/internal/auth/authenticator.go /home/developer/auth-service/internal/auth/authenticator.go
COPY home/developer/auth-service/internal/auth/authenticator_test.go /home/developer/auth-service/internal/auth/authenticator_test.go

# Set working directory
WORKDIR /home/developer/auth-service