from strands import Agent
from strands_mlx import MLXModel
from strands_tools import calculator

agent = Agent(
    model=MLXModel(
        model_id="mlx-community/Qwen3-1.7B-4bit", adapter_path="cagataydev/qwen3-strands-adapter"
    ),
    tools=[calculator],
)

agent("what is 2+2?")
# while True:
#     agent(input("\n# "))
