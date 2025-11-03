#!/usr/bin/env python3
"""
Example 10: Advanced Training Configuration
===========================================

Advanced training with YAML configuration and custom parameters.

Requirements:
    pip install strands-mlx strands-agents strands-agents-tools

Usage:
    python examples/10_advanced_training.py
"""

import yaml
from strands import Agent
from strands_tools import calculator

from strands_mlx import MLXModel, MLXSessionManager, dataset_splitter, mlx_trainer


def main():
    print("⚙️ Advanced Training Configuration Example\n")

    # Step 1: Collect training data
    print("📊 Step 1: Collecting training data...\n")

    session = MLXSessionManager(session_id="advanced_training", storage_dir="./advanced_data")

    model = MLXModel(model_id="mlx-community/Qwen3-1.7B-4bit")
    agent = Agent(
        model=model, tools=[calculator, dataset_splitter, mlx_trainer], session_manager=session
    )

    # Collect more training samples
    print("💬 Having training conversations...\n")
    for i in range(5):
        agent(f"What is {i+10} * {i+5}?")

    print("\n✅ Training data saved to: ./advanced_data/advanced_training.jsonl\n")

    # Step 2: Create advanced YAML config
    print("⚙️  Step 2: Creating advanced training config...\n")

    config = {
        "model": "mlx-community/Qwen3-1.7B-4bit",
        "data": "./advanced_data/advanced_training",
        "adapter_path": "./adapter_advanced",
        "iters": 500,
        "learning_rate": 1e-5,
        "batch_size": 1,
        "lora_parameters": {"rank": 8, "scale": 16.0, "dropout": 0.05},
        "lr_schedule": {"name": "cosine_decay", "warmup": 100, "arguments": [1e-5, 500, 1e-7]},
        "optimizer": "adamw",
        "optimizer_config": {"weight_decay": 0.01},
        "grad_checkpoint": True,
        "max_seq_length": 2048,
    }

    # Save config to YAML
    config_path = "./advanced_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"✅ Config saved to: {config_path}\n")

    # Display config
    print("📋 Training Configuration:\n")
    print(yaml.dump(config, default_flow_style=False))

    # Step 3: Split dataset
    print("\n📂 Step 3: Splitting dataset...\n")
    split_result = agent.tool.dataset_splitter(input_path="./advanced_data/advanced_training.jsonl")
    print(split_result)

    # Step 4: Train with config file
    print("\n🚀 Step 4: Training with advanced config...\n")
    train_result = agent.tool.mlx_trainer(action="train", config=config_path)
    print(train_result)

    print("\n📊 Config features:")
    print("  - ✅ Cosine decay learning rate schedule")
    print("  - ✅ AdamW optimizer with weight decay")
    print("  - ✅ LoRA rank 8 with dropout")
    print("  - ✅ Gradient checkpointing")
    print("  - ✅ Extended sequence length (2048)\n")

    print("✅ Example complete!")


if __name__ == "__main__":
    main()
