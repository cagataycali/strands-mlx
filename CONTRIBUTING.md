# Contributing to strands-mlx

Thank you for your interest in contributing to strands-mlx! This guide will help you get started.

## 🎯 Types of Contributions

We welcome:

- 🐛 **Bug fixes** - Help us squash issues
- ✨ **New features** - Expand MLX capabilities
- 📚 **Documentation** - Improve guides and examples
- 🧪 **Tests** - Increase test coverage
- 🎨 **Examples** - Share training workflows and use cases
- 🔧 **Tools** - Add new MLX tools for agents

---

## 🚀 Development Setup

### Prerequisites

- **macOS or Linux** (Apple Silicon recommended for MLX)
- **Python 3.10-3.13**
- **uv** package manager (recommended)

### Clone and Setup

```bash
# Clone the repository
git clone https://github.com/cagataycali/strands-mlx.git
cd strands-mlx

# Create virtual environment
uv venv --python 3.13
source .venv/bin/activate  # On macOS/Linux

# Install dependencies
uv pip install -e ".[all]"  # All features (vision, audio, video)
# OR
uv pip install -e ".[dev]"  # Development dependencies only
```

### Development Dependencies

```bash
# Install dev tools
uv pip install ruff black pytest pytest-asyncio
```

---

## 🧪 Running Tests

We use **pytest** for all tests. Tests are located in the `tests/` directory.

### Run All Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_inference.py -v
pytest tests/test_training.py -v
pytest tests/test_vision.py -v
```

### Test Categories

- **Inference Tests** (`test_inference.py`) - Basic agent + calculator
- **Training Tests** (`test_training.py`) - Data collection, validation, splitting
- **Vision Tests** (`test_vision.py`, `test_vision_invoke.py`) - Image analysis
- **Audio Tests** (`test_audio.py`) - Audio transcription
- **Video Tests** (`test_video_single.py`) - Video analysis
- **Runtime Tests** (`test_mlx_invoke.py`) - Model switching

### Test Media

Test media files are in `test_media/`:
- `sample_image.jpg` - HTTP status code visualization
- `audio_task_completed.wav` - Speech sample
- `sample_video.mp4` - Video sample

See `test_media/README.md` for regeneration scripts.

---

## 🎨 Code Style

We use **ruff** for linting and **black** for formatting.

### Before Committing

```bash
# Auto-fix linting issues
ruff check strands_mlx/ --fix

# Format code
black strands_mlx/ tests/

# Check for remaining issues
ruff check strands_mlx/
```

### Configuration

- Ruff config: `pyproject.toml` (line-length: 88)
- Black config: `pyproject.toml` (line-length: 88)

**Note:** CI automatically fixes linting/formatting issues and pushes back to your PR!

---

## 📝 Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feat/my-feature
# OR
git checkout -b fix/my-bugfix
```

### 2. Make Changes

- Write tests for new features
- Update documentation if needed
- Follow code style guidelines

### 3. Test Locally

```bash
# Run tests
pytest tests/ -v

# Check code style
ruff check strands_mlx/
black --check strands_mlx/
```

### 4. Commit with Semantic Messages

We use **semantic versioning** based on commit messages:

```bash
# Minor version bump (0.2.4 → 0.3.0)
git commit -m "feat: add new training feature"

# Patch version bump (0.2.4 → 0.2.5)
git commit -m "fix: resolve memory leak in mlx_trainer"
git commit -m "docs: improve README examples"
git commit -m "test: add vision model tests"
git commit -m "refactor: simplify session manager"

# Major version bump (0.2.4 → 1.0.0)
git commit -m "feat!: breaking API change"
# OR
git commit -m "feat: major change

BREAKING CHANGE: removed old API"
```

### 5. Push and Create PR

```bash
git push origin feat/my-feature
```

Create a pull request on GitHub with:
- Clear description of changes
- Link to related issues (if any)
- Test results (if applicable)

---

## 🐛 Reporting Issues

### Bug Reports

Include:
- **Description** - What happened vs. expected
- **Reproduction** - Minimal code to reproduce
- **Environment** - Python version, macOS version, hardware
- **Error logs** - Full traceback if available

### Feature Requests

Include:
- **Use case** - Why is this needed?
- **Proposed solution** - How should it work?
- **Alternatives** - Other approaches considered

---

## 🤝 Community Guidelines

- **Be respectful** - Treat everyone with kindness
- **Be constructive** - Provide helpful feedback
- **Be patient** - Maintainers are volunteers
- **Be collaborative** - Work together toward solutions

---

## 🎓 Learning Resources

- [Strands Agents Docs](https://strandsagents.com)
- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [MLX-LM Repository](https://github.com/ml-explore/mlx-examples/tree/main/llms)
- [mlx-community Models](https://huggingface.co/mlx-community)

---

## 💬 Questions?

- **GitHub Issues** - Report bugs or request features
- **GitHub Discussions** - Ask questions and share ideas
- **Pull Requests** - Submit code contributions

---

**Thank you for contributing to strands-mlx!** 🚀

Every contribution, no matter how small, helps make local MLX agents better for everyone.
