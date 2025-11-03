#!/usr/bin/env python3
"""
Example 03: Vision - Image Analysis
===================================

Analyze images with MLX vision model.

Requirements:
    pip install "strands-mlx[vision]" strands-agents

Usage:
    python examples/03_vision_image.py
"""

from strands import Agent

from strands_mlx import MLXVisionModel


def main():
    print("🖼️ Vision: Image Analysis Example\n")

    # Create MLX vision model
    print("📦 Loading model: mlx-community/Qwen2-VL-2B-Instruct-4bit...")
    model = MLXVisionModel(model_id="mlx-community/Qwen2-VL-2B-Instruct-4bit")

    # Create agent
    agent = Agent(model=model)

    # Analyze image
    image_path = "../test_media/sample_image.jpg"
    print(f"\n💬 Query: Describe the image at {image_path}\n")

    response = agent(f"Describe: <image>{image_path}</image>")

    print(f"🤖 Response:\n{response!s}\n")

    # Detailed analysis
    print("\n💬 Query: What HTTP status codes are shown?\n")
    response = agent(f"What HTTP status codes are shown in this image? <image>{image_path}</image>")

    print(f"🤖 Response:\n{response}\n")
    print("✅ Example complete!")


if __name__ == "__main__":
    main()
