from strands import Agent
from strands_mlx import MLXVisionModel

# Create vision model
model = MLXVisionModel(model_id="mlx-community/Qwen2-VL-2B-Instruct-4bit")
agent = Agent(model=model)

# Test image path
image_path = "./test_media/sample_image.jpg"

# Test 1: Simple image description
print("=== Test 1: Image description ===")
result = agent(f"What do you see in this image? <image>{image_path}</image>")
print(f"Result: {result}\n")

# Test 2: Detailed analysis
print("=== Test 2: Detailed analysis ===")
result = agent(
    f"Analyze this image in detail and describe the status codes shown: <image>{image_path}</image>"
)
print(f"Result: {result}\n")

# Test 3: Without explicit image tag (just path in message)
print("=== Test 3: Direct path ===")
result = agent(f"Describe {image_path}")
print(f"Result: {result}\n")
