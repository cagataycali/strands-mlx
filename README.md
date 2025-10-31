# strands-mlx

**Run and train AI agents locally on Apple Silicon**

MLX model provider for [Strands Agents](https://strandsagents.com) - inference, tool calling, and LoRA fine-tuning.

---

## Installation

**Requirements:** Python ≤3.13, macOS/Linux

**Quick install with uv (recommended):**
```bash
uv venv --python 3.13 && source .venv/bin/activate && uv pip install strands-agents strands-agents-tools strands-mlx
```

**Or with pip:**
```bash
pip install strands-mlx
```

---

## Quick Start

```python
from strands import Agent
from strands_mlx import MLXModel
from strands_tools import calculator # pip install strands-agents-tools

model = MLXModel(model_id="mlx-community/Qwen3-1.7B-4bit")
agent = Agent(model=model, tools=[calculator])

agent("What is 29 * 42?")
```

---

## Dynamic Model Invocation

**mlx_invoke** - Call MLX models as tools with custom configs:

```python
from strands import Agent
from strands_mlx import mlx_invoke

agent = Agent(tools=[mlx_invoke])

# Agent can invoke different MLX models dynamically
agent("Use mlx_invoke to ask Qwen3-1.7B to calculate 29 * 42 with calculator")

# Direct tool call with custom parameters
agent.tool.mlx_invoke(
    prompt="Explain quantum computing",
    system_prompt="You are a physics expert.",
    model_id="mlx-community/Qwen3-1.7B-4bit",
    params={"temperature": 0.7, "max_tokens": 2000},
    tools=["calculator"]
)
```

---

## Training Workflow

### 1. Collect → 2. Train → 3. Deploy

```python
from strands import Agent
from strands_mlx import MLXModel, MLXSessionManager, mlx_trainer

# 1. Collect training data (auto-exports to JSONL)
model = MLXModel(model_id="mlx-community/Qwen3-1.7B-4bit")
session = MLXSessionManager(session_id="my_training_data")
agent = Agent(model=model, session_manager=session)

agent("teach me about neural networks")
agent("how does backpropagation work?")
# Saved to: ~/.strands/mlx_training_data/my_training_data.jsonl

# 2. Train with LoRA
mlx_trainer(
    action="train",
    model="mlx-community/Qwen3-1.7B-4bit",
    data="~/.strands/mlx_training_data/my_training_data.jsonl",
    adapter_path="./trained_adapter",
    iters=1000
)

# 3. Deploy trained model
trained = MLXModel(model_id="mlx-community/Qwen3-1.7B-4bit", adapter_path="./trained_adapter")
agent = Agent(model=trained)
```

---

## Features

- **Local inference** - Run on Apple Silicon with unified memory
- **Tool calling** - Native Strands tools support  
- **Streaming** - Token-by-token generation
- **Dynamic invocation** - mlx_invoke tool for runtime model switching
- **Training pipeline** - Conversation → JSONL → LoRA → Deploy
- **1000+ models** - mlx-community quantized models (4-bit, 8-bit)

---

## Configuration

**Model:**
```python
model = MLXModel(model_id="mlx-community/Qwen3-1.7B-4bit", adapter_path="./adapters")
model.update_config(params={"temperature": 0.7, "max_tokens": 2000})
```

**Training:**
```python
mlx_trainer(
    action="train",
    model="...",
    data="path.jsonl",
    adapter_path="./adapters",
    batch_size=4,
    iters=1000,
    learning_rate=1e-5,
    lora_rank=8
)
```

---

## Development

```bash
git clone https://github.com/cagataycali/strands-mlx.git
cd strands-mlx
pip install -e ".[dev]"
python3 test.py
```

---

## Resources

- [Strands Agents](https://strandsagents.com)
- [MLX](https://ml-explore.github.io/mlx/) by Apple ML Research
- [mlx-community models](https://huggingface.co/mlx-community)

---

## Citation

```bibtex
@software{strands_mlx2025,
  author = {Cagatay Cali},
  title = {strands-mlx: MLX Model Provider for Strands Agents},
  year = {2025},
  url = {https://github.com/cagataycali/strands-mlx}
}
```

**Apache 2 License** | Built with MLX, MLX-LM, and Strands Agents
