#!/usr/bin/env python3
"""
Interactive agent for strands-mlx testing.

Usage:
    python agent.py "What is Strands Agents?"
    python agent.py "Explain training" --model mlx-community/Qwen3-1.7B-4bit
    python agent.py "Task" --adapter cagataydev/qwen3-strands-adapter
"""
import sys
import argparse
from strands import Agent
from strands_mlx import MLXModel
from strands_tools import calculator, current_time


def main():
    parser = argparse.ArgumentParser(description="Run strands-mlx agent")
    parser.add_argument("task", help="Task for the agent")
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen3-1.7B-4bit",
        help="Model to use (default: mlx-community/Qwen3-1.7B-4bit)",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="Optional adapter path (HF repo or local path)",
    )

    args = parser.parse_args()

    # Model configuration
    print(f"🤖 Loading model: {args.model}")
    if args.adapter:
        print(f"📦 With adapter: {args.adapter}")
        model = MLXModel(model_id=args.model, adapter_path=args.adapter)
    else:
        model = MLXModel(model_id=args.model)

    # Create agent
    agent = Agent(model=model, tools=[calculator, current_time])

    # Run task
    print(f"\n📋 Task: {args.task}\n")
    response = agent(args.task)
    print(f"\n✅ Response:\n{response!s}")


if __name__ == "__main__":
    main()
