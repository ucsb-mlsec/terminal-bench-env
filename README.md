# TermiGen: High-Fidelity Environments for Terminal Agents

Official implementation and datasets for **TermiGen**, a framework for training robust terminal agents through verified environments and error-correction trajectories.

📄 **Paper:** [TermiGen: High-Fidelity Environment and Robust Trajectory Synthesis for Terminal Agents](https://arxiv.org/abs/XXXX.XXXXX)  
🤖 **Model:** [TermiGen-32B](https://huggingface.co/UCSB-SURFI/TerminGen-32B)  
🧪 **Benchmark:** [TerminalBench](https://github.com/laude-institute/terminal-bench)

---

## 🎯 What's Included

This repository provides:

1. **3,500+ Verified Docker Environments** - Executable tasks across 11 categories (as ZIP archive)
2. **BashAgent** - Minimal ReAct-style agent implementation (`bash_agent.py`)


---

## 📊 Performance

Our TermiGen-32B achieves:

| Benchmark | Pass@1 |
|-----------|--------|
| TerminalBench 1.0 | **31.3%** |
| TerminalBench 2.0 | **18.0%** |

- **+26.8%** absolute improvement over base Qwen2.5-Coder-32B
- **+11.3%** absolute improvement over o4-mini with Codex CLI

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

# Extract environments
unzip termigen_env.zip -d environments/
```

### Step 2: Deploy TermiGen Model
```bash
# Install vLLM
pip install vllm

# Deploy model (requires GPU)
vllm serve ucsb-mlsec/TermiGen-32B \
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
export MODEL_NAME="ucsb-mlsec/TermiGen-32B"

# Run agent on TerminalBench 1.0 (example task: hello-world)
tb run --dataset terminal-bench-core==0.1.1 --agent-import-path bash_agent:BashAgent --task-id hello-world --log-level debug
```

---

## 📁 Repository Structure
```
terminal-bench-env/
├── README.md                      # This file
├── bash_agent.py                  # Minimal ReAct agent implementation
├── requirements.txt               # Python dependencies
├── termigen_environments.zip      # 3,500+ Docker tasks (download separately)
└── tasks/                         # Extracted environments (after unzip)
    ├── build_compilation/
    ├── system_admin/
    ├── security_forensics/
    └── ...
```

---

## 🔧 Environment Details

### Using Individual Tasks

After extracting `termigen_environments.zip`:
```bash
tb run --agent claude-code --model anthropic/claude-sonnet-4-5-20250929 --dataset-path tasks --task-id XXX --log-level debug --n-concurrent 1
```

### Using Full Dataset
```bash
tb run --agent claude-code --model anthropic/claude-sonnet-4-5-20250929 --dataset terminal-bench-core==0.1.1 --dataset-path tasks --log-level debug --n-concurrent 4
```


### Task Structure

Each task follows TerminalBench 1.0 format, it contains:
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

---

## 🤖 BashAgent

The `bash_agent.py` provides a minimal terminal agent implementation. 

- `model_endpoint`: Model API URL (OpenAI-compatible)
- `model_name`: Model identifier  
- `max_episodes`: Max conversation turns (default: 1000)
- `temperature`: Sampling temperature (default: 0.6)
- `max_tokens`: Max tokens per generation (default: 5000)
- `command_duration_sec`: Command timeout in seconds (default: 10.0).

### Model Compatibility

✅ **Fully supported**: Qwen2.5-Coder, TermiGen models (support "tool" message role)

⚠️ **Requires modification**: Models without "tool" role support need line 411 changed:
```python
# Change from:
{"role": "tool", "content": observation}

# To:
{"role": "user", "content": f"Observation: {observation}"}
```

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
