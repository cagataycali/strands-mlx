"""MLX Vision LoRA Training Tool for Strands Agents.

This tool wraps mlx-vlm's LoRA training functionality for fine-tuning vision/audio/video models
on training data collected from Strands conversations.
"""

import traceback
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from strands import tool

logger = logging.getLogger(__name__)


@tool
def mlx_vision_trainer(
    action: str,
    model: str = "mlx-community/Qwen2-VL-2B-Instruct-4bit",
    dataset: str = None,
    adapter_path: str = "./vision_adapters",
    learning_rate: float = 1e-5,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    num_epochs: int = 3,
    batch_size: int = 1,
    image_resize_shape: Optional[tuple] = None,
    apply_chat_template: bool = True,
    split: str = "train",
    max_steps: Optional[int] = None,
    save_steps: int = 100,
    logging_steps: int = 10,
) -> Dict[str, Any]:
    """MLX Vision LoRA training tool for fine-tuning vision/audio/video models.

    This tool trains vision language models using mlx-vlm's LoRA implementation.
    It supports Qwen2-VL, LLaVA, Gemma3-V, and other vision models.

    Key Differences from mlx_trainer:
    - Expects dataset with "messages" and "images" columns
    - Applies LoRA to language_model specifically (freezes vision tower)
    - Uses image processor for pixel value handling
    - Supports audio and video through mlx-vlm's infrastructure

    Args:
        action: Action to perform:
            - "train": Train vision model with LoRA
            - "test": Evaluate model on test set (not yet implemented)
            - "train_and_test": Train then evaluate (not yet implemented)
        model: Model path or HuggingFace repo (e.g., "mlx-community/Qwen2-VL-2B-Instruct-4bit")
        dataset: Path to HuggingFace dataset or local dataset directory
                Must have "messages" and "images" columns
                Format: {"messages": [...], "images": [...]}
        adapter_path: Directory to save/load adapter weights
        learning_rate: Learning rate for optimizer
        lora_rank: LoRA rank parameter
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        num_epochs: Number of training epochs
        batch_size: Training batch size (1 recommended for vision models)
        image_resize_shape: Optional tuple (height, width) to resize images
        apply_chat_template: Whether to apply chat template to messages
        split: Dataset split to use (default: "train")
        max_steps: Maximum training steps (overrides num_epochs if set)
        save_steps: Save checkpoint every N steps
        logging_steps: Log metrics every N steps

    Returns:
        Dict containing status and training results

    Example Dataset Format:
        ```json
        {
            "messages": [
                {"role": "user", "content": "What's in this image?"},
                {"role": "assistant", "content": "I see a cat."}
            ],
            "images": ["path/to/image.jpg"]
        }
        ```

    Notes:
        - Requires mlx-vlm: pip install strands-mlx[vision]
        - Vision tower is frozen during training
        - LoRA adapters only applied to language_model
        - Batch size of 1 recommended for memory efficiency
        - Use image_resize_shape to prevent OOM on large images
    """
    try:
        # Check if mlx-vlm is available
        try:
            from mlx_vlm import load
            from mlx_vlm.utils import load_image_processor
            from mlx_vlm.trainer import Dataset, Trainer, save_adapter
            from mlx_vlm.trainer.utils import (
                apply_lora_layers,
                find_all_linear_names,
                get_peft_model,
            )
            from mlx_vlm.prompt_utils import apply_chat_template as vlm_apply_chat_template
            import mlx.optimizers as optim
            from datasets import load_dataset
        except ImportError:
            return {
                "status": "error",
                "content": [
                    {
                        "text": "❌ **mlx-vlm not installed!**\n\n"
                        "Vision model training requires mlx-vlm:\n"
                        "```bash\n"
                        "pip install strands-mlx[vision]\n"
                        "```"
                    }
                ],
            }

        # Validate action
        if action not in ["train", "test", "train_and_test"]:
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"Invalid action: {action}. Must be 'train', 'test', or 'train_and_test'."
                    }
                ],
            }

        if action in ["test", "train_and_test"]:
            return {
                "status": "error",
                "content": [
                    {
                        "text": "❌ Test/evaluation not yet implemented for vision models.\n"
                        "Use action='train' to train only."
                    }
                ],
            }

        # Validate dataset
        if dataset is None:
            return {
                "status": "error",
                "content": [
                    {
                        "text": "Dataset is required. Provide HuggingFace dataset name or local path.\n"
                        "Dataset must have 'messages' and 'images' columns."
                    }
                ],
            }

        logger.info(f"🚀 Starting MLX Vision LoRA training...")
        logger.info(f"📦 Model: {model}")
        logger.info(f"📊 Dataset: {dataset}")
        logger.info(f"💾 Adapters: {adapter_path}")
        logger.info(f"🔧 Config: lr={learning_rate}, epochs={num_epochs}, batch_size={batch_size}")
        logger.info(f"🎯 LoRA: rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}\n")

        # Load model and processor
        logger.info(f"Loading model from {model}...")
        vision_model, processor = load(model, processor_config={"trust_remote_code": True})
        config = vision_model.config.__dict__
        image_processor = load_image_processor(model)

        # Load dataset
        logger.info(f"Loading dataset from {dataset}...")
        hf_dataset = load_dataset(dataset, split=split)

        # Validate dataset columns
        if "messages" not in hf_dataset.column_names:
            return {
                "status": "error",
                "content": [
                    {
                        "text": "❌ Dataset must have a 'messages' column.\n"
                        'Format: {"messages": [{"role": "user", "content": "..."}]}'
                    }
                ],
            }

        if "images" not in hf_dataset.column_names:
            return {
                "status": "error",
                "content": [
                    {
                        "text": "❌ Dataset must have an 'images' column.\n"
                        'Format: {"images": ["path/to/image.jpg"]}'
                    }
                ],
            }

        # Apply chat template if requested
        if apply_chat_template:
            logger.info("Applying chat template to dataset...")

            def process_data(examples):
                import json

                if config["model_type"] == "pixtral":
                    conversations = vlm_apply_chat_template(
                        config=config,
                        processor=processor,
                        prompt=examples["messages"],
                        return_messages=True,
                    )
                    examples["messages"] = [
                        json.dumps(item, ensure_ascii=False) for item in conversations
                    ]
                else:
                    examples["messages"] = vlm_apply_chat_template(
                        config=config,
                        processor=processor,
                        prompt=examples["messages"],
                        return_messages=True,
                    )
                return examples

            hf_dataset = hf_dataset.map(process_data)

        # Create dataset wrapper
        vision_dataset = Dataset(
            hf_dataset,
            config,
            processor,
            image_processor=image_processor,
            image_resize_shape=image_resize_shape,
        )

        # Setup LoRA
        if os.path.exists(adapter_path) and os.path.exists(
            os.path.join(adapter_path, "adapter_config.json")
        ):
            logger.info(f"Resuming from adapter path {adapter_path}")
            vision_model = apply_lora_layers(vision_model, adapter_path)
        else:
            logger.info("Setting up LoRA...")
            # Find linear layers in language_model only (freeze vision tower)
            list_of_modules = find_all_linear_names(vision_model.language_model)
            vision_model = get_peft_model(
                vision_model,
                list_of_modules,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )

        # Setup optimizer
        logger.info("Setting up optimizer...")
        optimizer = optim.Adam(learning_rate=learning_rate)

        # Setup trainer
        logger.info("Setting up trainer...")
        trainer = Trainer(vision_model, optimizer)

        # Set model to training mode
        vision_model.train()

        # Training loop
        logger.info("Starting training...")
        total_steps = max_steps if max_steps else len(vision_dataset) * num_epochs
        step = 0

        os.makedirs(adapter_path, exist_ok=True)

        from tqdm import tqdm

        with tqdm(total=total_steps, desc="Training") as pbar:
            for epoch in range(num_epochs):
                for batch_idx in range(0, len(vision_dataset), batch_size):
                    # Get batch
                    batch = vision_dataset[batch_idx : batch_idx + batch_size]

                    # Training step
                    loss = trainer.step(batch)

                    step += 1
                    pbar.update(1)

                    # Logging
                    if step % logging_steps == 0:
                        logger.info(f"Step {step}/{total_steps} | Loss: {loss:.4f}")
                        pbar.set_postfix({"loss": f"{loss:.4f}"})

                    # Save checkpoint
                    if step % save_steps == 0:
                        logger.info(f"Saving checkpoint at step {step}...")
                        save_adapter(adapter_path, vision_model, optimizer, step, loss)

                    # Max steps check
                    if max_steps and step >= max_steps:
                        break

                if max_steps and step >= max_steps:
                    break

        # Final save
        logger.info("Saving final adapter...")
        save_adapter(adapter_path, vision_model, optimizer, step, loss)

        # Build success response
        adapter_path_obj = Path(adapter_path)
        adapter_file = adapter_path_obj / "adapters.safetensors"
        config_file = adapter_path_obj / "adapter_config.json"

        result_text = f"✅ **MLX Vision LoRA training complete!**\n\n"
        result_text += "**📊 Training Summary:**\n"
        result_text += f"- Model: {model}\n"
        result_text += f"- Steps: {step}\n"
        result_text += f"- Epochs: {num_epochs}\n"
        result_text += f"- Final loss: {loss:.4f}\n"
        result_text += f"- Batch size: {batch_size}\n"
        result_text += f"- Learning rate: {learning_rate}\n"
        result_text += f"- LoRA rank: {lora_rank}\n\n"

        result_text += "**💾 Saved Files:**\n"
        if adapter_file.exists():
            size_mb = adapter_file.stat().st_size / (1024 * 1024)
            result_text += f"- Adapters: {adapter_file} ({size_mb:.1f}MB)\n"
        if config_file.exists():
            result_text += f"- Config: {config_file}\n"

        result_text += f"\n**🎯 Next steps:**\n"
        result_text += f"```python\n"
        result_text += f"# Load vision model with trained adapter\n"
        result_text += f"from strands_mlx import MLXVisionModel\n\n"
        result_text += f"model = MLXVisionModel(\n"
        result_text += f'    model_id="{model}",\n'
        result_text += f'    adapter_path="{adapter_path}"\n'
        result_text += f")\n"
        result_text += f"```"

        return {"status": "success", "content": [{"text": result_text}]}

    except Exception as e:
        error_trace = traceback.format_exc()
        return {
            "status": "error",
            "content": [{"text": f"❌ **Vision training failed:**\n\n```\n{error_trace}\n```"}],
        }
