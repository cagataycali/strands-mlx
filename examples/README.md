# strands-mlx Examples

Complete, runnable examples showcasing strands-mlx capabilities.

## xamples Overview

| Example | Description | Features |
|---------|-------------|----------|
| **01_basic_inference.py** | Simple text generation | MLXModel, basic agent |
| **02_tool_usage.py** | Agent with calculator tool | Tool calling, reasoning |
| **03_vision_image.py** | Image analysis | MLXVisionModel, image input |
| **04_vision_audio.py** | Audio transcription | Audio processing |
| **05_vision_video.py** | Video understanding | Video analysis |
| **06_training_pipeline.py** | Complete training flow | Data collection → training → usage |
| **07_adapter_loading.py** | Load trained adapters | HuggingFace adapter integration |
| **08_runtime_switching.py** | Switch models at runtime | mlx_invoke tool |
| **09_session_manager.py** | Collect training data | MLXSessionManager, JSONL export |
| **10_advanced_training.py** | Advanced training config | YAML config, custom parameters |

---

## Quick Start

### Prerequisites

```bash
# Install strands-mlx with all features
pip install "strands-mlx[all]"

# Or with uv
uv pip install "strands-mlx[all]"
```

### Run Examples

```bash
# Basic examples (no setup required)
python examples/01_basic_inference.py
python examples/02_tool_usage.py

# Vision examples (requires test media)
python examples/03_vision_image.py
python examples/04_vision_audio.py
python examples/05_vision_video.py

# Training examples
python examples/06_training_pipeline.py
python examples/09_session_manager.py
python examples/10_advanced_training.py
```

---

## Example Details

### 01 - Basic Inference
Simple text generation with Qwen3-1.7B model.
```bash
python examples/01_basic_inference.py
```

### 02 - Tool Usage
Agent with calculator tool performing mathematical reasoning.
```bash
python examples/02_tool_usage.py
```

### 03 - Vision: Image Analysis
Analyze images with Qwen2-VL vision model.
```bash
# Uses test_media/sample_image.jpg
python examples/03_vision_image.py
```

### 04 - Vision: Audio Transcription
Transcribe audio with Gemma3n audio model.
```bash
# Uses test_media/audio_task_completed.wav
python examples/04_vision_audio.py
```

### 05 - Vision: Video Understanding
Analyze video content with Qwen2-VL.
```bash
# Uses test_media/sample_video.mp4
python examples/05_vision_video.py
```

### 06 - Training Pipeline
Complete training workflow: collect → split → train → use.
```bash
python examples/06_training_pipeline.py
# Creates ./training_data/ and ./adapter_example/
```

### 07 - Adapter Loading
Load and use trained LoRA adapters from HuggingFace.
```bash
python examples/07_adapter_loading.py
```

### 08 - Runtime Model Switching
Switch between models at runtime using mlx_invoke.
```bash
python examples/08_runtime_switching.py
```

### 09 - Session Manager
Collect training data from conversations automatically.
```bash
python examples/09_session_manager.py
# Creates ./session_data/example_session.jsonl
```

### 10 - Advanced Training
Advanced training with YAML configuration.
```bash
python examples/10_advanced_training.py
# Creates ./advanced_config.yaml and trains with custom settings
```

---

## Common Patterns

### Pattern 1: Basic Agent Setup
```python
from strands import Agent
from strands_mlx import MLXModel

model = MLXModel(model_id="mlx-community/Qwen3-1.7B-4bit")
agent = Agent(model=model)
response = agent("Your prompt here")
```

### Pattern 2: Agent with Tools
```python
from strands import Agent
from strands_mlx import MLXModel
from strands_tools import calculator

model = MLXModel(model_id="mlx-community/Qwen3-1.7B-4bit")
agent = Agent(model=model, tools=[calculator])
response = agent("Calculate 15 * 7")
```

### Pattern 3: Vision Model
```python
from strands import Agent
from strands_mlx import MLXVisionModel

model = MLXVisionModel(model_id="mlx-community/Qwen2-VL-2B-Instruct-4bit")
agent = Agent(model=model)
response = agent("Describe: <image>photo.jpg</image>")
```

### Pattern 4: Training Data Collection
```python
from strands import Agent
from strands_mlx import MLXModel, MLXSessionManager

session = MLXSessionManager(session_id="training", storage_dir="./data")
model = MLXModel(model_id="mlx-community/Qwen3-1.7B-4bit")
agent = Agent(model=model, session_manager=session)

# Conversations auto-saved to JSONL
agent("First conversation")
agent("Second conversation")
```

### Pattern 5: Load Trained Adapter
```python
from strands import Agent
from strands_mlx import MLXModel

# Local adapter
model = MLXModel(
    model_id="mlx-community/Qwen3-1.7B-4bit",
    adapter_path="./my_adapter"
)

# HuggingFace adapter
model = MLXModel(
    model_id="mlx-community/Qwen3-1.7B-4bit",
    adapter_path="username/adapter-repo"
)

agent = Agent(model=model)
```

---

## Troubleshooting

### Out of Memory
```python
# Reduce batch size and sequence length
config = {
    "batch_size": 1,
    "max_seq_length": 1024,
    "grad_checkpoint": True
}
```

### Missing Test Media
```bash
# Generate test media files
cd test_media
python generate_test_media.py
```

### Model Download Issues
```bash
# Pre-download models
huggingface-cli download mlx-community/Qwen3-1.7B-4bit
```

---

## Learn More

- [strands-mlx Documentation](https://github.com/cagataycali/strands-mlx)
- [Strands Agents](https://strandsagents.com)
- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [mlx-community Models](https://huggingface.co/mlx-community)
