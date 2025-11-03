"""Video analysis tests for strands-mlx"""

import pytest
from pathlib import Path

# Test media path
TEST_VIDEO = Path("./test_media/sample_video.mp4")


def test_video_analysis():
    """Test video analysis functionality"""
    from strands import Agent
    from strands_mlx import MLXVisionModel

    model = MLXVisionModel(
        model_id="mlx-community/Qwen2-VL-2B-Instruct-4bit",
        params={"resize_shape": (1024, 1024), "max_tokens": 500},
    )
    agent = Agent(model=model)

    response = agent(f"Describe what happens in this video: <video>{TEST_VIDEO}</video>")
    print(f"\n🎥 Video analysis response:\n{response}\n")

    # Assertions
    assert response is not None
    assert len(str(response)) > 0

    # Check for video-related content
    response_text = str(response).lower()
    assert len(response_text) > 10  # Should have substantial description


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
