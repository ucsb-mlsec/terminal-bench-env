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
    apt-get install -y opam m4 && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y ocaml && \
    rm -rf /var/lib/apt/lists/*

RUN opam init -y --disable-sandboxing && \
    opam switch create 4.14.0 && \
    eval $(opam env) && \
    opam install -y dune

RUN mkdir -p /workspace/mathlib/core /workspace/mathlib/wrapper

COPY mathlib/core/numbers.ml /workspace/mathlib/core/numbers.ml
COPY mathlib/core/numbers.mli /workspace/mathlib/core/numbers.mli
COPY mathlib/wrapper/api.ml /workspace/mathlib/wrapper/api.ml
COPY mathlib/test_suite.ml /workspace/mathlib/test_suite.ml
COPY mathlib/dune /workspace/mathlib/dune

WORKDIR /workspace