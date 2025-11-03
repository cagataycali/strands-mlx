"""Strands MLX Model Provider for Apple Silicon."""

from strands_mlx.mlx_model import MLXModel
from strands_mlx.mlx_session_manager import MLXSessionManager
from strands_mlx.tools import mlx_invoke, mlx_trainer

# Vision support (optional dependency)
try:
    from strands_mlx.mlx_vision_model import MLXVisionModel
    from strands_mlx.tools.mlx_vision_invoke import mlx_vision_invoke
    from strands_mlx.tools.mlx_vision_trainer import mlx_vision_trainer

    __all__ = [
        "MLXModel",
        "MLXSessionManager",
        "mlx_trainer",
        "mlx_invoke",
        "MLXVisionModel",
        "mlx_vision_invoke",
        "mlx_vision_trainer",
    ]
except ImportError:
    # mlx-vlm not installed
    __all__ = ["MLXModel", "MLXSessionManager", "mlx_trainer", "mlx_invoke"]
