#!/usr/bin/env python3
"""Test audio analysis with MLXVisionModel."""

from strands import Agent
from strands_mlx import MLXVisionModel

# Use Gemma3n - audio-capable model
model = MLXVisionModel(
    model_id="mlx-community/gemma-3n-E2B-it-5bit", params={"temperature": 0.7, "max_tokens": 1000}
)

agent = Agent(model=model)

# Test audio file
audio_path = "./test_media/sample_audio.wav"

print("=" * 50)
print("🎵 AUDIO ANALYSIS TEST")
print("=" * 50)
print(f"\nModel: gemma-3n-E2B-it-5bit")
print(f"Audio: {audio_path}\n")

# Test 1: Basic transcription
print("\n" + "=" * 50)
print("Test 1: Basic Audio Transcription")
print("=" * 50)
result = agent(f"Transcribe this audio: <audio>{audio_path}</audio>")
print(f"Result: {result}")

print("\n" + "=" * 50)
print("✅ Audio test completed successfully!")
print("=" * 50)
