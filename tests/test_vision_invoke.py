from strands import Agent
from strands_mlx import MLXModel, mlx_vision_invoke

# Create a text-only agent with vision invoke tool
model = MLXModel(model_id="mlx-community/Qwen3-1.7B-4bit")
agent = Agent(model=model, tools=[mlx_vision_invoke])

print("=== Test 1: Direct tool call - single image ===")
result = agent.tool.mlx_vision_invoke(
    prompt="What do you see in this image?",
    images=["./test_media/sample_image.jpg"],
)
print(f"Status: {result['status']}")
for content in result["content"]:
    print(content["text"])
print()

print("=== Test 2: Agent naturally uses vision tool ===")
result = agent(
    "Use mlx_vision_invoke to analyze the image at ./test_media/sample_image.jpg and describe what you see"
)
print(f"Result: {result}\n")

print("=== Test 3: Multi-image comparison ===")
# Use the same image twice for testing
result = agent.tool.mlx_vision_invoke(
    prompt="Describe this image",
    images=["./test_media/sample_image.jpg"],
)
print(f"Status: {result['status']}")
for content in result["content"]:
    print(content["text"])
print()

print("=== Test 4: Custom parameters ===")
result = agent.tool.mlx_vision_invoke(
    prompt="Briefly describe this image in one sentence.",
    images=["./test_media/sample_image.jpg"],
    params={"temperature": 0.3, "max_tokens": 100},
    system_prompt="You are a concise image analyst.",
)
print(f"Status: {result['status']}")
for content in result["content"]:
    print(content["text"])
