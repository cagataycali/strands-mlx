# strands-mlx

**Run and train AI agents locally on Apple Silicon**

MLX model provider for [Strands Agents](https://strandsagents.com) - inference, tool calling, and LoRA fine-tuning.

---

## Installation

**Requirements:** Python ≤3.13, macOS/Linux

```bash
# Basic (text models)
pip install strands-mlx

# With vision/audio/video
pip install "strands-mlx[vision]"

# Quick setup with uv (recommended)
uv venv --python 3.13 && source .venv/bin/activate && uv pip install strands-agents strands-mlx
```

---

## Quick Start

```python
from strands import Agent
from strands_mlx import MLXModel

model = MLXModel("mlx-community/Qwen3-1.7B-4bit")
agent = Agent(model=model)

agent("What is 29 * 42?")
```

---

## Vision Models

**Multi-modal inference** - Images, audio, video:

```python
from strands_mlx import MLXVisionModel, mlx_vision_invoke

# Direct vision model
model = MLXVisionModel("mlx-community/Qwen2-VL-2B-Instruct-4bit")
agent = Agent(model=model)

agent("Describe: <image>photo.jpg</image>")
agent("Transcribe: <audio>speech.wav</audio>")  # Qwen2-Audio
agent("What happens: <video>clip.mp4</video>")

# Or as dynamic tool
agent = Agent(model=MLXModel("Qwen3-1.7B-4bit"), tools=[mlx_vision_invoke])
agent("Use mlx_vision_invoke to analyze chart.png")
```

---

## Training Pipeline

**1. Collect → 2. Train → 3. Deploy**

```python
from strands_mlx import MLXModel, MLXSessionManager, mlx_trainer, mlx_vision_trainer

# Text models
session = MLXSessionManager(session_id="training_data")
agent = Agent(model=MLXModel("Qwen3-1.7B-4bit"), session_manager=session)
agent("teach me about neural networks")  # Auto-saves to JSONL

mlx_trainer(action="train", model="Qwen3-1.7B-4bit", 
            data="~/.strands/mlx_training_data/training_data.jsonl",
            adapter_path="./adapter", iters=1000)

trained = MLXModel("Qwen3-1.7B-4bit", adapter_path="./adapter")

# Vision models
mlx_vision_trainer(action="train", model="Qwen2-VL-2B-Instruct-4bit",
                   dataset="my_dataset",  # HF dataset: messages + images
                   adapter_path="./vision_adapter", num_epochs=3)
```

---

## Dynamic Invocation

```python
from strands_mlx import mlx_invoke

agent = Agent(tools=[mlx_invoke])

# Runtime model switching
agent.tool.mlx_invoke(
    prompt="Explain quantum computing",
    model_id="Qwen3-1.7B-4bit",
    params={"temperature": 0.7, "max_tokens": 2000},
    tools=["calculator"]
)
```

---

## Features

- **Local inference** - Apple Silicon unified memory
- **Multi-modal** - Vision, audio, video (Qwen2-VL, Qwen2-Audio, LLaVA)
- **Tool calling** - Native Strands integration
- **Streaming** - Token-by-token generation
- **Training** - LoRA fine-tuning (text + vision)
- **1000+ models** - mlx-community quantized models

---

## Configuration

```python
# Model params
model = MLXModel("Qwen3-1.7B-4bit", adapter_path="./adapters")
model.update_config(params={"temperature": 0.7, "max_tokens": 2000})

# Training params
mlx_trainer(action="train", batch_size=4, iters=1000, 
            learning_rate=1e-5, lora_rank=8)
```

---

## Resources

- [Strands Agents](https://strandsagents.com)
- [MLX](https://ml-explore.github.io/mlx/)
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
