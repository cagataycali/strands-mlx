"""Strands MLX Model Provider for Apple Silicon."""

from strands_mlx.mlx_model import MLXModel
from strands_mlx.mlx_session_manager import MLXSessionManager
from strands_mlx.tools import (
    dataset_splitter,
    mlx_invoke,
    mlx_trainer,
    validate_training_data,
)

# Vision support (optional dependency)
try:
    from strands_mlx.mlx_vision_model import MLXVisionModel
    from strands_mlx.tools import mlx_vision_invoke, mlx_vision_trainer

    __all__ = [
        "MLXModel",
        "MLXSessionManager",
        "mlx_trainer",
        "mlx_invoke",
        "dataset_splitter",
        "validate_training_data",
        "MLXVisionModel",
        "mlx_vision_invoke",
        "mlx_vision_trainer",
    ]
except ImportError:
    # mlx-vlm not installed
    __all__ = [
        "MLXModel",
        "MLXSessionManager",
        "mlx_trainer",
        "mlx_invoke",
        "dataset_splitter",
        "validate_training_data",
    ]
