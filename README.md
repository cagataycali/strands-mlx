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

## Vision Language Models

**Installation:**
```bash
uv pip install "strands-mlx[vision]"  # Includes mlx-vlm, pillow, torch, soundfile
```

**MLXVisionModel** - Analyze images, audio, and video:

```python
from strands import Agent
from strands_mlx import MLXVisionModel

model = MLXVisionModel(model_id="mlx-community/Qwen2-VL-2B-Instruct-4bit")
agent = Agent(model=model)

# Images
agent("Describe what you see: <image>path/to/photo.jpg</image>")

# Audio (requires audio-capable model like Qwen2-Audio)
agent("Transcribe: <audio>speech.wav</audio>")

# Video (auto frame extraction)
agent("What happens in: <video>clip.mp4</video>")
```

**mlx_vision_invoke** - Dynamic vision model calls:

```python
from strands import Agent
from strands_mlx import MLXModel, mlx_vision_invoke

# Text-only agent with vision tool
agent = Agent(
    model=MLXModel("mlx-community/Qwen3-1.7B-4bit"),
    tools=[mlx_vision_invoke]
)

# Agent dynamically invokes vision models
agent("Use mlx_vision_invoke to analyze this chart: chart.png")
```

**Multi-modal examples:**

```python
# Multi-image analysis
agent.tool.mlx_vision_invoke(
    prompt="Compare these images",
    images=["before.jpg", "after.jpg"]
)

# Audio transcription
agent.tool.mlx_vision_invoke(
    prompt="Transcribe this audio",
    audio=["speech.wav"],
    model_id="mlx-community/Qwen2-Audio-7B-Instruct"
)

# Video analysis
agent.tool.mlx_vision_invoke(
    prompt="What happens in this video?",
    video=["recording.mp4"]
)
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

### Text Models: 1. Collect → 2. Train → 3. Deploy

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

### Vision Models: Training with Images/Audio/Video

```python
from strands_mlx import mlx_vision_trainer

# Train on dataset with images
mlx_vision_trainer(
    action="train",
    model="mlx-community/Qwen2-VL-2B-Instruct-4bit",
    dataset="my_vision_dataset",  # HF dataset with "messages" + "images" columns
    adapter_path="./vision_adapter",
    num_epochs=3,
    lora_rank=8,
    image_resize_shape=(1024, 1024)  # Prevent OOM
)

# Deploy trained vision model
from strands_mlx import MLXVisionModel
trained_vision = MLXVisionModel(
    model_id="mlx-community/Qwen2-VL-2B-Instruct-4bit",
    adapter_path="./vision_adapter"
)
```

---

## Features

- **Local inference** - Run on Apple Silicon with unified memory
- **Vision, audio, video** - Multi-modal models (Qwen2-VL, Qwen2-Audio, LLaVA, Gemma3-V)
- **Tool calling** - Native Strands tools support  
- **Streaming** - Token-by-token generation
- **Dynamic invocation** - mlx_invoke tool for runtime model switching
- **Training pipeline** - Text & vision model fine-tuning with LoRA
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
