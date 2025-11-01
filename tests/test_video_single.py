#!/usr/bin/env python3
"""Test video analysis with MLXVisionModel."""

from strands import Agent
from strands_mlx import MLXVisionModel

# Create MLX vision model
model = MLXVisionModel(
    model_id="mlx-community/Qwen2-VL-2B-Instruct-4bit",
    params={"resize_shape": (1024, 1024), "max_tokens": 500},
)

agent = Agent(model=model)

# Test video path
video_path = "./test_media/sample_video.mp4"

# Test video analysis
print(f"🎥 Analyzing video: {video_path}\n")
result = agent(f"Describe what happens in this video: <video>{video_path}</video>")
print(f"\n✅ Result: {result}")
