#!/usr/bin/env python3
"""
Example 06: Complete Training Pipeline
======================================

Full training workflow: collect → split → train → use.

Requirements:
    pip install strands-mlx strands-agents strands-agents-tools

Usage:
    python examples/06_training_pipeline.py
"""
import json
import os
import re
import time

from strands import Agent
from strands_tools import calculator

from strands_mlx import MLXModel, MLXSessionManager, dataset_splitter, mlx_trainer


def main():
    print("🎓 Complete Training Pipeline Example\n")

    # Step 1: Collect training data
    print("📊 Step 1: Collecting training data...\n")

    session = MLXSessionManager(session_id="example_training", storage_dir="./training_data")

    model = MLXModel(model_id="mlx-community/Qwen3-1.7B-4bit")
    agent = Agent(
        model=model, tools=[calculator, dataset_splitter, mlx_trainer], session_manager=session
    )

    # Have conversations to collect data (need at least 10 for proper split)
    print("💬 Having training conversations...\n")
    agent("What is 15 * 7?")
    agent("Calculate 29 * 42")
    agent("What is the square root of 144?")
    agent("What is 8 + 12?")
    agent("Calculate 50 / 5")
    agent("What is 2 to the power of 8?")
    agent("What is 100 - 37?")
    agent("Calculate 12 * 11")
    agent("What is 225 divided by 15?")
    agent("What is 7 + 8 + 9?")

    print("\n✅ Training data saved to: ./training_data/example_training.jsonl\n")

    # Step 2: Split dataset
    print("📂 Step 2: Splitting dataset...\n")
    split_result = agent.tool.dataset_splitter(input_path="./training_data/example_training.jsonl")
    print(f"Split result: {split_result['status']}\n")

    if split_result["status"] != "success":
        print("❌ Dataset splitting failed! Cannot proceed with training.\n")
        return

    # Step 3: Train with LoRA
    print("🚀 Step 3: Training LoRA adapter...\n")
    train_result = agent.tool.mlx_trainer(
        action="train",
        config={
            "model": "mlx-community/Qwen3-1.7B-4bit",
            "data": "./training_data",  # Points to directory with train/valid/test.jsonl
            "adapter_path": "./adapter_example",
            "iters": 30,
            "learning_rate": 1e-5,
            "batch_size": 1,
        },
    )

    if train_result.get("status") != "success":
        print("❌ Failed to start training!\n")
        print(f"Error: {train_result}")
        return

    # Extract task ID from result
    task_id = None
    for content in train_result.get("content", []):
        text = content.get("text", "")
        if "Task ID:" in text:
            # Extract task ID from text like "**Task ID:** `mlx_train_xxxxx`"
            match = re.search(r"mlx_train_\w+", text)
            if match:
                task_id = match.group(0)
                break

    if task_id:
        print(f"Training started with task ID: {task_id}\n")

        # Wait for training to complete
        print("⏳ Waiting for training to complete...\n")

        while True:
            # Call the tool (logs to JSONL)
            agent.tool.mlx_trainer(action="status", task_id=task_id)

            # Read the last line from the JSONL file
            jsonl_path = "./training_data/example_training.jsonl"
            try:
                with open(jsonl_path, "r") as f:
                    lines = f.readlines()
                    if lines:
                        last_line = lines[-1]
                        last_entry = json.loads(last_line)
                        response_text = last_entry.get("text", "").lower()

                        # Use regex to extract status field
                        status_match = re.search(r"\*\*status:\*\*\s*(\w+)", response_text)

                        if status_match:
                            status_value = status_match.group(1)

                            if status_value == "completed":
                                print("✅ Training completed!\n")
                                break
                            elif status_value == "failed":
                                print("❌ Training failed!\n")
                                print(response_text[:500])
                                return
                            else:
                                print("⏳ Training in progress...")
                        elif "not found" in response_text:
                            # Task cleaned up after completion
                            print("✅ Training completed (task cleaned up)!\n")
                            break
                        else:
                            print("⏳ Training in progress...")
            except Exception as e:
                print(f"⚠️ Error reading status: {e}")
                print("⏳ Continuing to poll...")

            time.sleep(10)

        # Step 4: Use trained model
        print("🎯 Step 4: Loading trained model...\n")

        # Verify adapter exists
        adapter_path = "./adapter_example"
        if not os.path.exists(adapter_path):
            print(f"❌ Adapter not found at {adapter_path}\n")
            print("Training may have completed but adapter was not saved.\n")
            return

        trained_model = MLXModel(
            model_id="mlx-community/Qwen3-1.7B-4bit", adapter_path=adapter_path
        )
        trained_agent = Agent(model=trained_model, tools=[calculator])

        print("💬 Testing trained model...\n")
        response = trained_agent("What is 20 * 30?")
        print(f"🤖 Response: {response}\n")

        print("✅ Complete pipeline finished!")
    else:
        print("❌ Could not extract task ID from training result")


if __name__ == "__main__":
    main()
