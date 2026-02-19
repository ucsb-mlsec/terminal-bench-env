FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /solution /config

# Create the sampling configuration files
RUN echo '{\n\
  "service_name": "all-services",\n\
  "default_sampling_probability": 0.05,\n\
  "error_sampling_probability": 1.0,\n\
  "critical_path_sampling_probability": 1.0\n\
}' > /solution/sampling_config.json

RUN echo '{\n\
  "service_name": "all-services",\n\
  "default_sampling_probability": 0.05,\n\
  "error_sampling_probability": 1.0,\n\
  "critical_path_sampling_probability": 1.0\n\
}' > /config/api-gateway-sampling.json

RUN echo '{\n\
  "service_name": "all-services",\n\
  "default_sampling_probability": 0.05,\n\
  "error_sampling_probability": 1.0,\n\
  "critical_path_sampling_probability": 1.0\n\
}' > /config/order-service-sampling.json

RUN echo '{\n\
  "service_name": "all-services",\n\
  "default_sampling_probability": 0.05,\n\
  "error_sampling_probability": 1.0,\n\
  "critical_path_sampling_probability": 1.0\n\
}' > /config/payment-service-sampling.json

WORKDIR /solution