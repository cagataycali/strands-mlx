"""MLX LoRA Training Tool for Strands Agents.

Config-driven training with internal background execution management.
"""

import json
import os
import threading
import time
import traceback
import types
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from strands import tool

# Global registry for background training tasks
_TRAINING_TASKS = {}
_TASK_LOCK = threading.Lock()


class TrainingTask:
    """Background training task."""

    def __init__(self, task_id: str, config: Dict[str, Any]):
        self.task_id = task_id
        self.config = config
        self.status = "pending"
        self.thread = None
        self.result = None
        self.error = None
        self.start_time = None
        self.end_time = None
        self.stop_flag = threading.Event()

    def start(self):
        """Start training in background thread."""
        self.status = "running"
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._run_training, daemon=True)
        self.thread.start()

    def _run_training(self):
        """Execute training (runs in background thread)."""
        try:
            result = _execute_training_sync(self.config, self.stop_flag)
            self.result = result
            self.status = "completed" if result["status"] == "success" else "failed"
        except Exception as e:
            self.error = traceback.format_exc()
            self.status = "failed"
        finally:
            self.end_time = time.time()

    def stop(self):
        """Signal training to stop."""
        self.stop_flag.set()
        self.status = "stopped"

    def get_info(self) -> Dict[str, Any]:
        """Get task information."""
        info = {
            "task_id": self.task_id,
            "status": self.status,
            "config": self.config,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }

        if self.start_time and self.end_time:
            info["duration_seconds"] = self.end_time - self.start_time
        elif self.start_time:
            info["duration_seconds"] = time.time() - self.start_time

        if self.result:
            info["result"] = self.result
        if self.error:
            info["error"] = self.error

        return info


def _load_config(config: Union[str, Dict, None]) -> Dict[str, Any]:
    """Load config from YAML file or dict.

    Args:
        config: Path to YAML file or dict with config

    Returns:
        Config dictionary
    """
    if config is None:
        return {}

    if isinstance(config, dict):
        return config

    if isinstance(config, str):
        # Try JSON parsing first (in case dict was serialized)
        try:
            parsed = json.loads(config)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

        # Try YAML file
        config_path = Path(config).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    raise TypeError(f"Config must be str (path) or dict, got {type(config)}")


def _merge_config(
    base_config: Dict[str, Any], overrides: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Merge config with overrides (deep merge for nested dicts).

    Args:
        base_config: Base configuration
        overrides: Override values

    Returns:
        Merged configuration
    """
    if not overrides:
        return base_config

    merged = base_config.copy()

    # Deep merge nested dicts
    for key in ["lora_parameters", "optimizer_config", "lr_schedule"]:
        if key in overrides:
            if key not in merged:
                merged[key] = {}
            if isinstance(merged[key], dict) and isinstance(overrides[key], dict):
                merged[key].update(overrides[key])
            else:
                merged[key] = overrides[key]

    # Merge top-level overrides (excluding already processed nested)
    for key, value in overrides.items():
        if key not in ["lora_parameters", "optimizer_config", "lr_schedule"]:
            merged[key] = value

    return merged


def _execute_training_sync(
    config: Dict[str, Any], stop_flag: Optional[threading.Event] = None
) -> Dict[str, Any]:
    """Execute MLX-LM training synchronously.

    Args:
        config: Training configuration
        stop_flag: Optional threading event to signal stop

    Returns:
        Training results
    """
    try:
        from mlx_lm.lora import run, CONFIG_DEFAULTS

        # Validate required fields
        if "data" not in config:
            raise ValueError("Config must include 'data' field")

        # Expand data path
        data_path = Path(config["data"]).expanduser().resolve()
        if not data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")

        # Start with CONFIG_DEFAULTS from mlx-lm
        full_config = CONFIG_DEFAULTS.copy()

        # Apply user config
        full_config.update(config)

        # Build args namespace for mlx-lm (exactly as lora.py expects)
        args = types.SimpleNamespace(**full_config)

        # Override data with expanded path
        args.data = str(data_path)

        # Set adapter_path default if not specified
        if not hasattr(args, "adapter_path") or args.adapter_path is None:
            args.adapter_path = "adapters"

        # Ensure adapter_path is expanded
        args.adapter_path = str(Path(args.adapter_path).expanduser().resolve())

        # Set tokenizers parallelism
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        # Log training start
        print(f"\n🚀 Starting MLX LoRA training...")
        print(f"📦 Model: {args.model}")
        print(f"📊 Data: {data_path}")
        print(f"💾 Adapters: {args.adapter_path}")
        print(f"🔧 Batch size: {args.batch_size}, Iters: {args.iters}, LR: {args.learning_rate}")

        if hasattr(args, "lora_parameters") and isinstance(args.lora_parameters, dict):
            lora = args.lora_parameters
            print(
                f"🎯 LoRA rank: {lora.get('rank', 8)}, scale: {lora.get('scale', 20.0)}, dropout: {lora.get('dropout', 0.0)}"
            )
            if "keys" in lora:
                print(f"🔑 LoRA keys: {lora['keys']}")

        if hasattr(args, "lr_schedule") and args.lr_schedule:
            print(f"📈 LR schedule: {args.lr_schedule.get('name', 'constant')}")

        if hasattr(args, "grad_checkpoint") and args.grad_checkpoint:
            print(f"💾 Gradient checkpointing: enabled")

        print()

        # Execute training via mlx-lm
        run(args)

        # Check if stopped
        if stop_flag and stop_flag.is_set():
            return {"status": "error", "content": [{"text": "Training stopped by user"}]}

        # Build success result
        adapter_path_obj = Path(args.adapter_path)
        adapter_file = adapter_path_obj / "adapters.safetensors"
        config_file = adapter_path_obj / "adapter_config.json"

        result_text = f"✅ **MLX LoRA training complete!**\n\n"
        result_text += "**📊 Training Summary:**\n"
        result_text += f"- Model: {args.model}\n"
        result_text += f"- Iterations: {args.iters}\n"
        result_text += f"- Batch size: {args.batch_size}\n"
        result_text += f"- Learning rate: {args.learning_rate}\n"

        if hasattr(args, "lora_parameters") and isinstance(args.lora_parameters, dict):
            lora = args.lora_parameters
            result_text += f"- LoRA rank: {lora.get('rank', 8)}\n"
            if "keys" in lora:
                result_text += f"- LoRA keys: {', '.join(lora['keys'])}\n"

        result_text += "\n**💾 Saved Files:**\n"
        if adapter_file.exists():
            size_mb = adapter_file.stat().st_size / (1024 * 1024)
            result_text += f"- Adapters: {adapter_file} ({size_mb:.1f}MB)\n"
        if config_file.exists():
            result_text += f"- Config: {config_file}\n"

        # List timestamped checkpoints
        checkpoints = sorted(adapter_path_obj.glob("*_adapters.safetensors"))
        if checkpoints:
            result_text += f"- Checkpoints: {len(checkpoints)} timestamped checkpoints\n"

        result_text += f"\n**🎯 Next steps:**\n"
        result_text += f"```python\n"
        result_text += f"from strands_mlx import MLXModel\n\n"
        result_text += f"model = MLXModel(\n"
        result_text += f'    model_id="{args.model}",\n'
        result_text += f'    adapter_path="{args.adapter_path}"\n'
        result_text += f")\n"
        result_text += f"```"

        return {"status": "success", "content": [{"text": result_text}]}

    except Exception as e:
        error_trace = traceback.format_exc()
        return {
            "status": "error",
            "content": [{"text": f"❌ **Training failed:**\n\n```\n{error_trace}\n```"}],
        }


@tool
def mlx_trainer(
    action: str,
    config: Union[str, Dict, None] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    task_id: Optional[str] = None,
) -> Dict[str, Any]:
    """MLX LoRA trainer - 100% wrapper for mlx-lm training with background execution.

    Supports all mlx-lm training features via config:
    - LoRA layer keys specification
    - Learning rate schedules (cosine_decay, linear, constant)
    - Optimizer configs (AdamW, Adam, SGD, Muon, Adafactor)
    - Gradient checkpointing
    - Gradient accumulation
    - Reporting to wandb/swanlab
    - HuggingFace dataset loading
    - Timestamped checkpoints

    Args:
        action: Action to perform:
            - "train": Start training in background (non-blocking)
            - "train_sync": Train in foreground (blocking)
            - "status": Check training task status
            - "stop": Stop background training
            - "list": List all training tasks
            - "get_result": Get final result of completed task
        config: Path to lora_config.yaml OR dict with full config.
            Supports all mlx-lm options from lora_config.yaml:
            - model: Model path/HF repo
            - data: Path to .jsonl or directory with train/valid/test.jsonl
            - train: Whether to train (default: true)
            - test: Whether to test after training
            - fine_tune_type: "lora", "dora", or "full"
            - optimizer: "adam", "adamw", "sgd", "muon", "adafactor"
            - optimizer_config: Nested config (betas, eps, weight_decay, etc.)
            - batch_size: Training batch size
            - iters: Number of training iterations
            - learning_rate: Learning rate
            - num_layers: Number of layers to fine-tune (-1 for all)
            - lora_parameters: Dict with rank, scale, dropout, keys
            - lr_schedule: Dict with name, warmup, arguments
            - val_batches: Validation batches (-1 for all)
            - test_batches: Test batches (-1 for all)
            - steps_per_report: Report loss every N steps
            - steps_per_eval: Evaluate every N steps
            - steps_per_save: Save checkpoint every N steps
            - max_seq_length: Maximum sequence length
            - adapter_path: Save/load path for adapters
            - grad_checkpoint: Use gradient checkpointing
            - grad_accumulation_steps: Gradient accumulation steps
            - resume_adapter_file: Resume from checkpoint
            - mask_prompt: Mask prompt in loss
            - report_to: "wandb", "swanlab", or "wandb,swanlab"
            - project_name: Project name for reporting
            - seed: Random seed
        config_overrides: Optional dict to override specific config values.
            Example: {"iters": 2000, "batch_size": 8}
        task_id: Task ID for background training (auto-generated if not provided)

    Returns:
        Dict containing status and training results or task info

    Examples:
        # Background training with YAML
        mlx_trainer(
            action="train",
            config="./lora_config.yaml",
            task_id="qwen3_training"
        )

        # Background training with dict
        mlx_trainer(
            action="train",
            config={
                "model": "mlx-community/Qwen3-1.7B-4bit",
                "data": "~/.strands/mlx_training_data/session.jsonl",
                "iters": 1000,
                "batch_size": 4,
                "learning_rate": 1e-5,
                "lora_parameters": {
                    "keys": ["self_attn.q_proj", "self_attn.v_proj"],
                    "rank": 8,
                    "scale": 20.0
                },
                "lr_schedule": {
                    "name": "cosine_decay",
                    "warmup": 100,
                    "arguments": [1e-5, 1000, 1e-7]
                }
            }
        )

        # Override config values
        mlx_trainer(
            action="train",
            config="./lora_config.yaml",
            config_overrides={"iters": 2000, "batch_size": 8}
        )

        # Foreground training (blocking)
        mlx_trainer(action="train_sync", config="./lora_config.yaml")

        # Check training status
        mlx_trainer(action="status", task_id="qwen3_training")

        # List all tasks
        mlx_trainer(action="list")

        # Stop training
        mlx_trainer(action="stop", task_id="qwen3_training")

        # Get result
        mlx_trainer(action="get_result", task_id="qwen3_training")
    """
    try:
        global _TRAINING_TASKS

        # Handle background training
        if action == "train":
            # Load and merge config
            base_config = _load_config(config)
            final_config = _merge_config(base_config, config_overrides)

            # Auto-set train=true when action is train
            if "train" not in final_config:
                final_config["train"] = True

            # Generate task_id if not provided
            if task_id is None:
                task_id = f"mlx_train_{uuid.uuid4().hex[:8]}"

            # Create and start task
            with _TASK_LOCK:
                if task_id in _TRAINING_TASKS:
                    return {
                        "status": "error",
                        "content": [{"text": f"Task {task_id} already exists"}],
                    }

                task = TrainingTask(task_id, final_config)
                _TRAINING_TASKS[task_id] = task
                task.start()

            return {
                "status": "success",
                "content": [
                    {
                        "text": f"🚀 **Training started in background!**\n\n"
                        f"**Task ID:** `{task_id}`\n\n"
                        f"**Model:** {final_config.get('model', 'N/A')}\n"
                        f"**Iterations:** {final_config.get('iters', 'N/A')}\n\n"
                        f"**Check status:**\n"
                        f"```python\n"
                        f'mlx_trainer(action="status", task_id="{task_id}")\n'
                        f"```\n\n"
                        f"**Agent remains responsive!**"
                    }
                ],
            }

        # Handle foreground training
        elif action == "train_sync":
            base_config = _load_config(config)
            final_config = _merge_config(base_config, config_overrides)
            return _execute_training_sync(final_config)

        # Handle status check
        elif action == "status":
            if task_id is None:
                return {
                    "status": "error",
                    "content": [{"text": "task_id required for status action"}],
                }

            with _TASK_LOCK:
                if task_id not in _TRAINING_TASKS:
                    return {"status": "error", "content": [{"text": f"Task {task_id} not found"}]}

                task = _TRAINING_TASKS[task_id]
                info = task.get_info()

            # Format status message
            status_text = f"**Task:** `{task_id}`\n"
            status_text += f"**Status:** {info['status']}\n"

            if info.get("duration_seconds"):
                duration = info["duration_seconds"]
                status_text += f"**Duration:** {duration:.1f}s\n"

            if info["status"] == "running":
                status_text += f"\n🏃 Training in progress...\n"
            elif info["status"] == "completed":
                status_text += f"\n✅ Training completed!\n"
            elif info["status"] == "failed":
                status_text += f"\n❌ Training failed\n"
                if info.get("error"):
                    status_text += f"\n**Error:**\n```\n{info['error'][:500]}\n```\n"

            return {"status": "success", "content": [{"text": status_text}]}

        # Handle get result
        elif action == "get_result":
            if task_id is None:
                return {
                    "status": "error",
                    "content": [{"text": "task_id required for get_result action"}],
                }

            with _TASK_LOCK:
                if task_id not in _TRAINING_TASKS:
                    return {"status": "error", "content": [{"text": f"Task {task_id} not found"}]}

                task = _TRAINING_TASKS[task_id]

            if task.status == "running":
                return {"status": "error", "content": [{"text": "Training still running"}]}

            if task.result:
                return task.result
            elif task.error:
                return {
                    "status": "error",
                    "content": [{"text": f"Training failed:\n```\n{task.error}\n```"}],
                }
            else:
                return {"status": "error", "content": [{"text": "No result available"}]}

        # Handle stop
        elif action == "stop":
            if task_id is None:
                return {
                    "status": "error",
                    "content": [{"text": "task_id required for stop action"}],
                }

            with _TASK_LOCK:
                if task_id not in _TRAINING_TASKS:
                    return {"status": "error", "content": [{"text": f"Task {task_id} not found"}]}

                task = _TRAINING_TASKS[task_id]
                task.stop()

            return {
                "status": "success",
                "content": [{"text": f"⏹️ Stopped training task: {task_id}"}],
            }

        # Handle list
        elif action == "list":
            with _TASK_LOCK:
                tasks_list = [task.get_info() for task in _TRAINING_TASKS.values()]

            if not tasks_list:
                return {"status": "success", "content": [{"text": "No training tasks"}]}

            list_text = f"**Training Tasks ({len(tasks_list)}):**\n\n"
            for info in tasks_list:
                list_text += f"- **{info['task_id']}**: {info['status']}"
                if info.get("duration_seconds"):
                    list_text += f" ({info['duration_seconds']:.1f}s)"
                list_text += "\n"

            return {"status": "success", "content": [{"text": list_text}]}

        else:
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"Invalid action: {action}. "
                        f"Must be 'train', 'train_sync', 'status', 'stop', 'list', or 'get_result'."
                    }
                ],
            }

    except Exception as e:
        error_trace = traceback.format_exc()
        return {
            "status": "error",
            "content": [{"text": f"❌ **Error:**\n\n```\n{error_trace}\n```"}],
        }
