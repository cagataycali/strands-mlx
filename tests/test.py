from strands import Agent
from strands_tools import calculator
from strands_mlx import MLXModel, MLXSessionManager

# Create session manager for training data export
session_manager = MLXSessionManager(
    session_id="training_session", storage_dir="./test_session", add_generation_prompt=True
)

model = MLXModel(model_id="mlx-community/qwen3-4b-4bit-DWQ")  # or mlx-community/Qwen3-1.7B-4bit
agent = Agent(model=model, tools=[calculator], session_manager=session_manager)

# Test 1: Simple query without tools
print("=== Test 1: Simple query ===")
agent("what is 2+2?")

print("\n=== Test 2: Calculator tool ===")
agent("use calculator to compute 15 * 7")
