"""Strands MLX Tools - Training and utilities for MLX models."""

from .mlx_invoke import mlx_invoke
from .mlx_trainer import mlx_trainer

# Vision tools (optional dependency)
try:
    from .mlx_vision_invoke import mlx_vision_invoke
    from .mlx_vision_trainer import mlx_vision_trainer

    __all__ = ["mlx_trainer", "mlx_invoke", "mlx_vision_invoke", "mlx_vision_trainer"]
except ImportError:
    # mlx-vlm not installed
    __all__ = ["mlx_trainer", "mlx_invoke"]
