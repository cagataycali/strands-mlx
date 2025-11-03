#!/usr/bin/env python3
"""
Example 09: Session Manager
===========================

Collect training data from conversations automatically.

Requirements:
    pip install strands-mlx strands-agents strands-agents-tools

Usage:
    python examples/09_session_manager.py
"""

from strands import Agent
from strands_tools import calculator

from strands_mlx import MLXModel, MLXSessionManager, dataset_splitter, mlx_trainer


def main():
    print("💾 Session Manager Example\n")

    # Create session manager
    print("📊 Creating session manager...")
    session = MLXSessionManager(session_id="example_session", storage_dir="./session_data")

    # Create agent with session manager
    print("📦 Loading model: mlx-community/Qwen3-1.7B-4bit...")
    model = MLXModel(model_id="mlx-community/Qwen3-1.7B-4bit")
    agent = Agent(
        model=model, tools=[calculator, dataset_splitter, mlx_trainer], session_manager=session
    )

    # Have conversations - all auto-saved to JSONL
    print("\n💬 Having conversations (auto-saved to JSONL)...\n")

    print("Query 1: What is 15 * 7?\n")
    response = agent("What is 15 * 7?")
    print(f"Response: {response!s}\n")

    print("Query 2: Calculate the square of 12\n")
    response = agent("Calculate the square of 12")
    print(f"Response: {response!s}\n")

    print("Query 3: What is 29 * 42?\n")
    response = agent("What is 29 * 42?")
    print(f"Response: {response!s}\n")

    # Data automatically saved
    print("✅ All conversations saved to: ./session_data/example_session.jsonl\n")

    # Split the dataset
    print("📂 Splitting dataset...\n")
    split_result = agent.tool.dataset_splitter(input_path="./session_data/example_session.jsonl")
    print(split_result)

    # Optional: Train with the data
    print("\n🚀 To train with this data, run:\n")
    print(
        """agent.tool.mlx_trainer(
    action="train",
    config={
        "model": "mlx-community/Qwen3-1.7B-4bit",
        "data": "./session_data/example_session",
        "adapter_path": "./adapter",
        "iters": 200
    }
)"""
    )

    print("\n✅ Example complete!")


if __name__ == "__main__":
    main()
