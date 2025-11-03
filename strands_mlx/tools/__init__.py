"""Strands MLX Tools - Training and utilities for MLX models."""

from .dataset_splitter import dataset_splitter
from .mlx_invoke import mlx_invoke
from .mlx_trainer import mlx_trainer
from .validate_training_data import validate_training_data

# Vision tools (optional dependency)
try:
    from .mlx_vision_invoke import mlx_vision_invoke
    from .mlx_vision_trainer import mlx_vision_trainer

    __all__ = [
        "mlx_trainer",
        "mlx_invoke",
        "dataset_splitter",
        "validate_training_data",
        "mlx_vision_invoke",
        "mlx_vision_trainer",
    ]
except ImportError:
    # mlx-vlm not installed
    __all__ = ["mlx_trainer", "mlx_invoke", "dataset_splitter", "validate_training_data"]
