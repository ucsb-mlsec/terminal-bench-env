# TermiGen: High-Fidelity Environments for Terminal Agents

Official implementation and datasets for **TermiGen**, a framework for training robust terminal agents through verified environments and error-correction trajectories.

📄 **Paper:** [TermiGen: High-Fidelity Environment and Robust Trajectory Synthesis for Terminal Agents](https://arxiv.org/abs/2602.07274)  
🤖 **Model:** [TermiGen-32B](https://huggingface.co/UCSB-SURFI/TerminGen-32B)  
🧪 **Benchmark:** [TerminalBench](https://github.com/laude-institute/terminal-bench)

---

## 📰 News

- [2026/02/23] TermiGen has integrated with the [harbor framework](https://harborframework.com/registry)! Simply install harbor and run `uvx harbor run -d termigen-environments@1.0`!
- [2026/02/10] Grateful to be selected as the #2 [🤗 Huggingface Daily Papers](https://huggingface.co/papers/2602.07274)!

## 🎯 What's Included

This repository provides:

1. **3,500+ Verified Docker Environments** - Executable tasks across 11 categories. The TerminalBench 1.0 format is provided as a ZIP archive (because some tasks involve GitHub repositories containing .git metadata), while the Harbor 2.0 format is provided directly as the `environments_harbor/` directory.
2. **BashAgent** - Minimal ReAct-style agent implementation:
   - `bash_agent.py` - For [TerminalBench](https://github.com/laude-institute/terminal-bench) (`tb`) framework
   - `bash_agent_harbor.py` - For [Harbor](https://github.com/laude-institute/harbor) evaluation framework


---

## 📊 Performance

Our TermiGen-32B achieves:

| Benchmark | Pass@1 |
|-----------|--------|
| TerminalBench 1.0 | **31.3%** |
| TerminalBench 2.0 | **19.3%** |
| SWE-Bench Verified | **21.4%** |
| SWE-Bench Pro | **~5.7%** |

- **+26.8%** absolute improvement over base Qwen2.5-Coder-32B on TerminalBench 1.0
- **+11.3%** absolute improvement over o4-mini with Codex CLI on TerminalBench 1.0

---

## 🗂️ Environment Categories

Our environments span **11 task categories** across **3 tiers**:

### Tier I: Infrastructure & Core Systems
- 🛠️ **Software Build & Compilation**: gcc, cmake, rustc, makefile debugging
- ⚙️ **System Administration & DevOps**: Docker, Kubernetes, systemd, nginx
- 🔐 **Security & Reverse Engineering**: Ghidra, Wireshark, gdb, Metasploit

### Tier II: Data & Algorithm Applications
- 📊 **Data Processing & ETL**: Spark, Kafka, Parquet, SQL transformations
- 🤖 **Machine Learning & MLOps**: PyTorch, CUDA, Hugging Face, model debugging
- 🧩 **Algorithms & Logic**: Graph algorithms, dynamic programming, search

### Tier III: Specialized Domains
- 💻 **Software Development**: React, Django, REST APIs, CI/CD
- 🧪 **Scientific Computing**: Bioconductor, RDKit, GROMACS, NumPy
- 🎮 **Interactive Environments**: WebSocket, SSH, Jupyter, REPL
- 🌐 **Distributed Computing**: MPI, OpenMP, Ray, SLURM
- 🔬 **Formal Verification**: Coq, Z3, OpenGL, Vulkan

**Statistics:**
- 420 unique command-line tools
- 16 functional domains
- Average task complexity: 25.5 turns, 8,722 tokens

---

## 🚀 Quick Start

### Step 1: Download Repository
```bash
# Clone repository
git clone https://github.com/ucsb-mlsec/terminal-bench-env.git
cd terminal-bench-env

# Extract environments (TerminalBench 1.0 format)
unzip termigen_env.zip -d environments/

# Harbor 2.0 format is already included as environments_harbor/
```

### Step 2: Deploy TermiGen Model
```bash
# Install vLLM
pip install vllm

# Deploy model (requires GPU)
vllm serve UCSB-SURFI/TermiGen-32B \
  --port 8000 \
  --tensor-parallel-size 4 \
  --dtype bfloat16
```

### Step 3: Run BashAgent on TerminalBench
```bash
# Install dependencies
pip install openai
pip install terminal-bench

# Set environment variables
export MODEL_ENDPOINT="http://localhost:8000/v1"
export MODEL_NAME="UCSB-SURFI/TermiGen-32B"

# Run agent on TerminalBench 1.0 (example task: hello-world)
tb run --dataset terminal-bench-core==0.1.1 --agent-import-path bash_agent:BashAgent --task-id hello-world --log-level debug
```

**Or with Harbor 2.0:**
```bash
pip install harbor

# Run agent on local tasks
harbor run -p environments_harbor/ \
  --agent-import-path bash_agent_harbor:BashAgent \
  -t hello-world --debug

# Run agent on Terminal-Bench 2.0 (downloads from registry)
harbor run -d terminal-bench@2.0 \
  --agent-import-path bash_agent_harbor:BashAgent \
  --n-concurrent 4 --n-attempts 5 -e e2b

# Run agent on SWE-Bench Verified
harbor run -d swebench-verified \
  --agent-import-path bash_agent_harbor:BashAgent \
  --n-concurrent 4 -e e2b
```

---

## 🔧 Environment Details

### Using Individual Tasks

After extracting environments (see Step 1):
```bash
# TerminalBench 1.0
tb run --agent claude-code --model anthropic/claude-sonnet-4-5-20250929 --dataset-path environments --task-id a_b_testing_models_medium --log-level debug --n-concurrent 1

# Harbor 2.0
harbor run -a claude-code -m anthropic/claude-sonnet-4-5-20250929 -p environments_harbor/ -t a_b_testing_models_medium --debug
```

### Using Full Dataset
```bash
# TerminalBench 1.0
tb run --agent claude-code --model anthropic/claude-sonnet-4-5-20250929 --dataset terminal-bench-core==0.1.1 --dataset-path environments --log-level debug --n-concurrent 2

# Harbor 2.0
harbor run -a claude-code -m anthropic/claude-sonnet-4-5-20250929 -p environments_harbor/ -n 2 --debug
# try using smaller n for stability.
```


### Task Structure

Each task is available in two formats:

**TerminalBench 1.0** (`termigen_env.zip`):
```
task_name/
├── task.yaml              # Task description and metadata
├── Dockerfile             # Environment specification  
├── docker-compose.yaml    # Container orchestration
├── run-tests.sh          # Test execution script
├── tests/                # Unit tests (pytest)
│   └── test_*.py
└── [task files]          # Source code, configs, data, git repos
```

**Harbor 2.0** (`environments_harbor/`):
```
task_name/
├── task.toml              # Task metadata
├── instruction.md         # Task description
├── environment/
│   └── Dockerfile         # Environment specification
├── tests/
│   └── test.sh           # Test script with reward logging
└── [task files]          # Source code, configs, data, git repos
```

---

## 📁 Repository Structure
```
terminal-bench-env/
├── README.md                      # This file
├── bash_agent.py                  # BashAgent for TerminalBench (tb) framework
├── bash_agent_harbor.py           # BashAgent for Harbor framework
├── termigen_env.zip               # 3,500+ Docker tasks (TerminalBench 1.0 format)
├── environments/                  # Extracted from termigen_env.zip (after unzip)
└── environments_harbor/           # 3,500+ Docker tasks in Harbor 2.0 format
```

---

## 🤖 BashAgent

Two implementations are provided for different evaluation frameworks:

### `bash_agent.py` (TerminalBench / `tb` framework)
- Interacts with the environment via `TmuxSession` (send_keys, incremental output)
- Supports asciinema recording and detailed failure mode tracking
- `command_duration_sec`: Command timeout in seconds (default: 10.0)

### `bash_agent_harbor.py` (Harbor framework)
- Async implementation using Harbor's `BaseEnvironment.exec()` API
- Compatible with Harbor's e2b, Docker, Daytona, and other environment backends
- `timeout_sec`: Command timeout in seconds (default: 120)

### Shared Configuration

Both agents accept the following via environment variables or constructor args:
- `MODEL_ENDPOINT`: Model API URL (OpenAI-compatible, default: `http://172.17.0.1:8001/v1`)
- `MODEL_NAME`: Model identifier
- `max_episodes`: Max conversation turns (default: 1000)
- `temperature`: Sampling temperature (default: 0.6)

### Model Compatibility

✅ **Fully supported**: Qwen2.5-Coder, TermiGen models (support "tool" message role)

⚠️ **Requires modification**: Models without "tool" role support need the observation message role changed:
```python
# Change from:
{"role": "tool", "content": observation}

# To:
{"role": "user", "content": f"Observation: {observation}"}
```

---

## 🔄 Reproducing Benchmark Results

### Terminal-Bench 2.0 (19.3%)

```bash
# 1. Deploy model
vllm serve UCSB-SURFI/TermiGen-32B --port 8001 --tensor-parallel-size 2

# 2. Run evaluation (89 tasks × 5 attempts)
MODEL_NAME=UCSB-SURFI/TermiGen-32B \
MODEL_ENDPOINT=http://127.0.0.1:8001/v1 \
harbor run -d terminal-bench@2.0 \
  --agent-import-path bash_agent_harbor:BashAgent \
  --job-name tb2_termigen \
  --n-concurrent 4 --n-attempts 5 -e e2b
```

### SWE-Bench Verified (21.4%)

```bash
# Same model deployment as above, then:
MODEL_NAME=UCSB-SURFI/TermiGen-32B \
MODEL_ENDPOINT=http://127.0.0.1:8001/v1 \
harbor run -d swebench-verified \
  --agent-import-path bash_agent_harbor:BashAgent \
  --job-name swebench_verified_termigen \
  --n-concurrent 4 --n-attempts 1 -e e2b
```

### SWE-Bench Pro (~5.7%)

```bash
MODEL_NAME=UCSB-SURFI/TermiGen-32B \
MODEL_ENDPOINT=http://127.0.0.1:8001/v1 \
harbor run -d swebenchpro \
  --agent-import-path bash_agent_harbor:BashAgent \
  --job-name swebench_pro_termigen \
  --n-concurrent 4 --n-attempts 1 -e e2b
```

> **Note:** These benchmarks use the [e2b](https://e2b.dev) cloud sandbox (`-e e2b`). You can also use local Docker (`-e docker`) but some SWE-Bench Pro tasks with large Docker images (>10GB) may fail on e2b due to template build limits.

<!-- ---

## 🔬 Key Features

### High-Fidelity Environments
- ✅ **100% executable**: Every environment verified through Docker build + unit tests
- ✅ **Diverse coverage**: 11 categories from low-level system ops to scientific computing
- ✅ **Automated validation**: Judge Agent ensures task solvability

### Error-Correction Training
- 🔄 **Active error injection**: 20% of trajectory steps contain intentional mistakes
- 🎯 **5 failure categories**: Analysis errors, command errors, hallucinations, requirement violations, verification failures
- 📈 **+43% improvement**: Error-correction training vs. standard SFT -->

---

## 📖 Citation

If you use TermiGen in your research, please cite:
```bibtex
@article{zhu2026termigen,
  title={TermiGen: High-Fidelity Environment and Robust Trajectory Synthesis for Terminal Agents},
  author={Zhu, Kaijie and Nie, Yuzhou and Li, Yijiang and Huang, Yiming and Wu, Jialian and Liu, Jiang and Sun, Ximeng and Yin, Zhenfei and Wang, Lun and Liu, Zicheng and Barsoum, Emad and Wang, William Yang and Guo, Wenbo},
  journal={arXiv preprint arXiv:2602.07274},
  url={https://arxiv.org/abs/2602.07274}, 
  year={2026}
}
```

---

## 🤝 Contributing

We welcome contributions! Please feel free to:
- Report bugs or request features via [GitHub Issues](https://github.com/ucsb-mlsec/terminal-bench-env/issues)
- Submit pull requests for bug fixes or improvements for our environments and tasks

---

## 📧 Contact

- **Lead Author**: Kaijie Zhu (kaijiezhu@ucsb.edu)
- **Issues**: [GitHub Issues](https://github.com/ucsb-mlsec/terminal-bench-env/issues)
- **Paper**: [arXiv](https://arxiv.org/abs/XXXX.XXXXX)

---

## 🙏 Acknowledgements

- **Base Model**: [Qwen2.5-Coder](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct) by Alibaba Cloud
- **Benchmark**: [TerminalBench](https://github.com/laude-institute/terminal-bench) by Laude Institute
- **Compute**: AMD MI325X GPUs
- **Institutions**: UC Santa Barbara, UC San Diego, AMD, University of Oxford, Google
