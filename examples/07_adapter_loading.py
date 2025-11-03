#!/usr/bin/env python3
"""
Example 07: Adapter Loading
===========================

Load and use trained LoRA adapters from local or HuggingFace.

Requirements:
    pip install strands-mlx strands-agents

Usage:
    python examples/07_adapter_loading.py
"""

from strands import Agent
from strands_tools import calculator

from strands_mlx import MLXModel


def main():
    print("🔌 Adapter Loading Example\n")

    # Example 1: Load local adapter
    print("📦 Example 1: Loading local adapter...\n")

    # Assumes you have a local adapter from training
    # If not, skip to Example 2
    try:
        model = MLXModel(
            model_id="mlx-community/Qwen3-1.7B-4bit", adapter_path="./adapter_example"  # Local path
        )
        agent = Agent(model=model, tools=[calculator])

        print("💬 Query with local adapter: What is 29 * 42?\n")
        response = agent("What is 29 * 42?")
        print(f"🤖 Response:\n{response!s}\n")
    except Exception as e:
        print(f"⚠️  Local adapter not found: {e}\n")

    # Example 2: Load HuggingFace adapter
    print("📦 Example 2: Loading HuggingFace adapter...\n")
    print("🌐 Downloading adapter from: cagataydev/qwen3-strands-adapter\n")

    model = MLXModel(
        model_id="mlx-community/Qwen3-1.7B-4bit",
        adapter_path="cagataydev/qwen3-strands-adapter",  # HuggingFace repo
    )
    agent = Agent(model=model, tools=[calculator])

    print("💬 Query with HuggingFace adapter: What is 29 * 42?\n")
    response = agent("What is 29 * 42?")
    print(f"🤖 Response:\n{response!s}\n")

    print("✅ Example complete!")


if __name__ == "__main__":
    main()
