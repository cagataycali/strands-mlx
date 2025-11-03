#!/usr/bin/env python3
"""
Example 08: Runtime Model Switching
===================================

Switch between different MLX models at runtime using mlx_invoke.

Requirements:
    pip install strands-mlx strands-agents strands-agents-tools

Usage:
    python examples/08_runtime_switching.py
"""

from strands import Agent
from strands_tools import calculator

from strands_mlx import MLXModel, mlx_invoke


def main():
    print("🔄 Runtime Model Switching Example\n")

    # Create agent with primary model
    print("📦 Loading primary model: mlx-community/Qwen3-1.7B-4bit...")
    model = MLXModel(model_id="mlx-community/Qwen3-1.7B-4bit")
    agent = Agent(model=model, tools=[calculator, mlx_invoke])

    # Use mlx_invoke
    print("\n💬 Query using mlx_invoke: What is 15 * 7?\n")
    response = agent("use mlx_invoke for 15 * 7")
    print(f"🤖 Response:\n{response}\n")

    # Note about mlx_invoke
    print("\n🔄 mlx_invoke tool usage...\n")
    print("   The mlx_invoke tool allows switching models at runtime")
    print("   It's available through agent.tool.mlx_invoke()")
    print("\n   Example:")
    print(
        """
   agent.tool.mlx_invoke(
       model_id="mlx-community/Qwen3-4B-4bit",
       prompt="What is 29 * 42?",
       tools=[calculator]
   )
   """
    )

    print("✅ Example complete!")


if __name__ == "__main__":
    main()
