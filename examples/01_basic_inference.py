#!/usr/bin/env python3
"""
Example 01: Basic Inference
============================

Simple text generation with MLX model.

Requirements:
    pip install strands-mlx strands-agents

Usage:
    python examples/01_basic_inference.py
"""

from strands import Agent

from strands_mlx import MLXModel


def main():
    print("🚀 Basic Inference Example\n")

    # Create MLX model
    print("📦 Loading model: mlx-community/Qwen3-1.7B-4bit...")
    model = MLXModel(model_id="mlx-community/Qwen3-1.7B-4bit")

    # Create agent
    agent = Agent(model=model)

    # Simple query
    print("\n💬 Query: Tell me about Apple Silicon in 2 sentences\n")
    response = agent("Tell me about Apple Silicon in 2 sentences")

    print(f"🤖 Response:\n{response!s}\n")
    print("✅ Example complete!")


if __name__ == "__main__":
    main()
