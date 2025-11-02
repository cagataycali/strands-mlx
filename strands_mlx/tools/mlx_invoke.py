"""MLX model invocation tool for Strands agents.

Invoke MLX models with custom settings, tools, and system prompts while
streaming through the parent agent's callback handler.
"""

import logging
from typing import Any, Dict, List, Optional

from strands import Agent, tool
from strands.telemetry.metrics import metrics_to_string

from strands_mlx import MLXModel

logger = logging.getLogger(__name__)


@tool
def mlx_invoke(
    prompt: str,
    system_prompt: str = "You are a helpful AI assistant.",
    model_id: str = "mlx-community/Qwen3-1.7B-4bit",
    adapter_path: Optional[str] = None,
    tools: Optional[List[str]] = None,
    params: Optional[Dict[str, Any]] = None,
    agent: Optional[Any] = None,
) -> Dict[str, Any]:
    """Invoke MLX model with custom configuration and parent agent's tools.

    This tool creates a temporary Agent instance with an MLX model, using the
    parent agent's callback handler for streaming and tool registry for tool
    access. Perfect for testing different MLX models or adapter configurations
    without modifying the main agent.

    How It Works:
    ------------
    1. Creates MLXModel instance with specified configuration
    2. Filters tools from parent agent's registry (if tools parameter provided)
    3. Creates temporary Agent with MLX model + filtered tools
    4. Passes parent agent's callback_handler for streaming
    5. Invokes agent with prompt and captures response + metrics
    6. Returns formatted result with model info and performance data

    Common Use Cases:
    ---------------
    - Test different MLX models (4-bit, 8-bit quantized)
    - Evaluate LoRA adapters on specific tasks
    - Compare model performance with different parameters
    - Run specialized models for domain-specific tasks
    - Quick prototyping with local models

    Args:
        prompt: The prompt to process with the MLX model.
        system_prompt: Custom system prompt for the agent (default: "You are a helpful AI assistant.").
        model_id: HuggingFace model ID or local path.
            Examples: "mlx-community/Qwen3-1.7B-4bit", "mlx-community/qwen3-4b-4bit-DWQ"
        adapter_path: Optional path to LoRA adapter directory.
            If provided, loads the adapter on top of the base model.
        tools: List of tool names to make available from parent agent.
            If not provided, inherits all tools from parent agent.
            Examples: ["calculator", "file_read", "retrieve"]
        params: MLX generation parameters.
            Supported keys:
            - temperature (float): Sampling temperature (default: 1.0)
            - max_tokens (int): Maximum tokens to generate (default: 3000)
            - top_p (float): Nucleus sampling parameter (default: 1.0)
            Example: {"temperature": 0.7, "max_tokens": 2000}
        agent: Parent agent (automatically provided by Strands framework).

    Returns:
        Dict containing status and response content:
        {
            "status": "success|error",
            "content": [
                {"text": "Response: The model's response"},
                {"text": "Model: Information about the MLX model used"},
                {"text": "Metrics: Performance metrics (tokens, latency)"}
            ]
        }

    Examples:
    --------
    # Basic usage with default model
    result = agent.tool.mlx_invoke(
        prompt="Explain quantum computing",
        system_prompt="You are a physics expert."
    )

    # Use different quantized model
    result = agent.tool.mlx_invoke(
        prompt="Write a poem about AI",
        system_prompt="You are a creative poet.",
        model_id="mlx-community/qwen3-4b-4bit-DWQ"
    )

    # Load model with LoRA adapter
    result = agent.tool.mlx_invoke(
        prompt="Explain this code",
        system_prompt="You are a code reviewer.",
        model_id="mlx-community/Qwen3-1.7B-4bit",
        adapter_path="./my_code_adapter"
    )

    # Custom generation parameters
    result = agent.tool.mlx_invoke(
        prompt="Analyze this data",
        system_prompt="You are a data analyst.",
        params={"temperature": 0.3, "max_tokens": 4000}
    )

    # Specific tools only
    result = agent.tool.mlx_invoke(
        prompt="Calculate 15 * 7 and explain",
        system_prompt="You are a math tutor.",
        tools=["calculator"]
    )

    Notes:
        - Requires mlx and mlx-lm packages installed
        - Only works on Apple Silicon (M1/M2/M3/M4)
        - Streaming uses parent agent's callback handler
        - Tools must exist in parent agent's tool registry
        - Adapter training can be done with mlx_trainer tool
        - Model and adapter are loaded fresh for each invocation
    """
    try:
        # Get tools and trace attributes from parent agent
        filtered_tools = []
        trace_attributes = {}
        extra_kwargs = {}

        if agent:
            trace_attributes = agent.trace_attributes
            # Pass parent's callback handler for streaming
            extra_kwargs["callback_handler"] = agent.callback_handler

            # Filter tools from parent agent
            if tools is not None:
                for tool_name in tools:
                    if tool_name in agent.tool_registry.registry:
                        filtered_tools.append(agent.tool_registry.registry[tool_name])
                    else:
                        logger.warning(
                            f"Tool '{tool_name}' not found in parent agent's tool registry"
                        )
            else:
                # Inherit all tools from parent
                filtered_tools = list(agent.tool_registry.registry.values())

        # Create MLX model with configuration
        model_config = {}
        if adapter_path:
            model_config["adapter_path"] = adapter_path
        if params:
            model_config["params"] = params

        logger.debug(f"🔄 Creating MLX model: {model_id}")
        mlx_model = MLXModel(model_id=model_id, **model_config)

        # Build model info string
        model_info = f"MLX Model: {model_id}"
        if adapter_path:
            model_info += f" (adapter: {adapter_path})"
        if params:
            model_info += f" | params: {params}"

        logger.debug(f"--- Model Info ---\n{model_info}")
        logger.debug(f"--- Input Prompt ---\n{prompt}\n")

        # Create temporary agent with MLX model
        logger.debug("🔄 Creating temporary agent with MLX model...")
        mlx_agent = Agent(
            model=mlx_model,
            messages=[],
            tools=filtered_tools,
            system_prompt=system_prompt,
            trace_attributes=trace_attributes,
            **extra_kwargs,
        )

        # Invoke agent with prompt
        result = mlx_agent(prompt)

        # Extract response
        assistant_response = str(result)
        logger.debug(f"\n--- Assistant Response ---\n{assistant_response.strip()}\n")

        # Extract metrics
        metrics_text = ""
        if result.metrics:
            metrics = result.metrics
            metrics_text = metrics_to_string(metrics)
            logger.debug(metrics_text)

        return {
            "status": "success",
            "content": [
                {"text": f"Response: {assistant_response}"},
                {"text": f"Model: {model_info}"},
                {"text": f"Metrics: {metrics_text}"},
            ],
        }

    except Exception as e:
        error_msg = f"Error in mlx_invoke tool: {str(e)}"
        logger.error(error_msg)
        return {
            "status": "error",
            "content": [{"text": error_msg}],
        }
