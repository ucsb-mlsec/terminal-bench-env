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

# Install dependencies for Lean 4
RUN apt-get update && \
    apt-get install -y curl git build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install elan (Lean version manager)
RUN curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y --default-toolchain leanprover/lean4:stable

# Add elan to PATH
ENV PATH="/root/.elan/bin:${PATH}"

# Create project directory structure
RUN mkdir -p /workspace/math_proofs

# Create lakefile.lean
RUN echo 'import Lake\nopen Lake DSL\n\npackage «math_proofs» where\n  version := v!"0.1.0"\n\nlean_lib «BasicArithmetic» where\n  roots := #[`BasicArithmetic]' > /workspace/math_proofs/lakefile.lean

# Create BasicArithmetic.lean with broken proofs
RUN echo 'import Mathlib.Data.Nat.Basic\nimport Mathlib.Tactic\n\ntheorem add_comm_custom (a b : Nat) : a + b = b + a := by\n  rw [Nat.add_comm a]\n  \ntheorem mul_distrib_custom (a b c : Nat) : a * (b + c) = a * b + a * c := by\n  rw [Nat.mul_add a b]\n  \ntheorem zero_add_custom (n : Nat) : 0 + n = n := by\n  exact Nat.zero_add' > /workspace/math_proofs/BasicArithmetic.lean

# Set working directory
WORKDIR /workspace/math_proofs

# Initialize Lake project and fetch dependencies
RUN lake update

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
