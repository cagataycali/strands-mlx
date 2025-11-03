#!/usr/bin/env python3
"""
Example 02: Tool Usage
======================

Agent with calculator tool demonstrating tool calling.

Requirements:
    pip install strands-mlx strands-agents strands-agents-tools

Usage:
    python examples/02_tool_usage.py
"""
from strands import Agent
from strands_tools import calculator

from strands_mlx import MLXModel


def main():
    print("🔧 Tool Usage Example\n")

    # Create MLX model
    print("📦 Loading model: mlx-community/Qwen3-1.7B-4bit...")
    model = MLXModel(model_id="mlx-community/Qwen3-1.7B-4bit")

    # Create agent with calculator tool
    agent = Agent(model=model, tools=[calculator])

    # Mathematical query requiring tool use
    print("\n💬 Query: What is 29 * 42?\n")
    response = agent("What is 29 * 42?")

    print(f"🤖 Response:\n{response!s}\n")

    # Complex calculation
    print("\n💬 Query: Calculate the area of a circle with radius 7\n")
    response = agent("Calculate the area of a circle with radius 7")

    print(f"🤖 Response:\n{response!s}\n")
    print("✅ Example complete!")


if __name__ == "__main__":
    main()
