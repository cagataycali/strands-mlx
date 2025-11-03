#!/usr/bin/env python3
"""
Example 05: Vision - Video Understanding
========================================

Analyze video content with MLX vision model.

Requirements:
    pip install "strands-mlx[vision]" strands-agents

Usage:
    python examples/05_vision_video.py
"""

from strands import Agent

from strands_mlx import MLXVisionModel


def main():
    print("🎥 Vision: Video Understanding Example\n")

    # Create MLX vision model
    print("📦 Loading model: mlx-community/Qwen2-VL-2B-Instruct-4bit...")
    model = MLXVisionModel(model_id="mlx-community/Qwen2-VL-2B-Instruct-4bit")

    # Create agent
    agent = Agent(model=model)

    # Analyze video
    video_path = "../test_media/sample_video.mp4"
    print("\n💬 Query: Describe what happens in the video\n")

    response = agent(f"Describe: <video>{video_path}</video>")

    print(f"🤖 Response:\n{response!s}\n")
    print("✅ Example complete!")


if __name__ == "__main__":
    main()
