"""Test mlx_invoke tool - invoke MLX models with custom settings."""

from strands import Agent
from strands_tools import calculator
from strands_mlx import mlx_invoke

# Create agent with mlx_invoke tool
agent = Agent(tools=[mlx_invoke, calculator])

print("=== Test 1: Basic mlx_invoke ===")
result = agent.tool.mlx_invoke(
    prompt="What is 2+2?",
    system_prompt="You are a helpful math assistant.",
    model_id="mlx-community/Qwen3-1.7B-4bit",
)
print(result)

print("\n=== Test 2: mlx_invoke with calculator tool ===")
result = agent.tool.mlx_invoke(
    prompt="Calculate 15 * 7 and explain the result",
    system_prompt="You are a math tutor.",
    model_id="mlx-community/Qwen3-1.7B-4bit",
    tools=["calculator"],
    params={"temperature": 0.7, "max_tokens": 2000},
)
print(result)

print("\n=== Test 3: Agent using mlx_invoke naturally ===")
response = agent(
    "Use mlx_invoke to ask the Qwen3-1.7B-4bit model to calculate 25 * 4 with the calculator tool"
)
print(response)
