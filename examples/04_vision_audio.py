#!/usr/bin/env python3
"""
Example 04: Vision - Audio Transcription
========================================

Transcribe audio with MLX audio model.

Requirements:
    pip install "strands-mlx[vision]" strands-agents

Usage:
    python examples/04_vision_audio.py
"""

from strands import Agent

from strands_mlx import MLXVisionModel


def main():
    print("🎵 Vision: Audio Transcription Example\n")

    # Create MLX audio model
    print("📦 Loading model: mlx-community/gemma-3n-E2B-it-5bit...")
    model = MLXVisionModel(model_id="mlx-community/gemma-3n-E2B-it-5bit")

    # Create agent
    agent = Agent(model=model)

    # Transcribe audio
    audio_path = "../test_media/audio_task_completed.wav"
    print(f"\n💬 Query: Transcribe the audio at {audio_path}\n")

    response = agent(f"Transcribe: <audio>{audio_path}</audio>")

    print(f"🤖 Response:\n{response!s}\n")
    print("✅ Example complete!")


if __name__ == "__main__":
    main()
