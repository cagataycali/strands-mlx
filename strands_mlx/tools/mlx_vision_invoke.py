"""Dynamic MLX vision model invocation tool for Strands agents.

Allows agents to invoke different vision models at runtime with custom configurations,
similar to mlx_invoke but for vision language models.
"""

import logging
from typing import Any, Dict, List, Optional

from strands import Agent, tool
from strands.telemetry.metrics import metrics_to_string

logger = logging.getLogger(__name__)


@tool
def mlx_vision_invoke(
    prompt: str,
    images: Optional[list[str]] = None,
    audio: Optional[list[str]] = None,
    video: Optional[list[str]] = None,
    system_prompt: str = "You are a helpful multimodal assistant.",
    model_id: str = "mlx-community/Qwen2-VL-2B-Instruct-4bit",
    adapter_path: Optional[str] = None,
    tools: Optional[List[str]] = None,
    params: Optional[Dict[str, Any]] = None,
    agent=None,
) -> Dict[str, Any]:
    """Invoke MLX vision model with images, audio, and video for dynamic multimodal analysis.

    This tool enables agents to dynamically invoke different vision language models
    at runtime with custom configurations, tools, and parameters.

    Args:
        prompt: Text prompt describing what to analyze
        images: Optional list of image URLs or file paths
        audio: Optional list of audio file paths (e.g., ["audio.wav", "speech.mp3"])
        video: Optional list of video file paths (e.g., ["clip.mp4"])
        system_prompt: System prompt defining the model's role and expertise
        model_id: HuggingFace model ID or local path (default: Qwen2-VL-2B-Instruct-4bit)
        adapter_path: Optional path to LoRA adapter weights
        tools: List of tool names from parent agent to make available
        params: Generation parameters dict (temperature, max_tokens, top_p, resize_shape)
        agent: Parent agent instance (auto-injected by Strands)

    Returns:
        Dict containing status and response content

    Examples:
        # Image analysis
        result = mlx_vision_invoke(
            prompt="What objects do you see?",
            images=["photo.jpg"]
        )

        # Audio transcription (requires audio-enabled model)
        result = mlx_vision_invoke(
            prompt="Transcribe this audio",
            audio=["speech.wav"],
            model_id="mlx-community/Qwen2-Audio-7B-Instruct"
        )

        # Video analysis
        result = mlx_vision_invoke(
            prompt="What happens in this video?",
            video=["clip.mp4"]
        )

        # Multi-modal: image + audio
        result = mlx_vision_invoke(
            prompt="Describe the scene and transcribe the speech",
            images=["scene.jpg"],
            audio=["speech.wav"]
        )

        # With resize to prevent OOM
        result = mlx_vision_invoke(
            prompt="Analyze this high-res image",
            images=["large.jpg"],
            params={"resize_shape": (1024, 1024)}
        )

    Notes:
        - Requires mlx-vlm: pip install strands-mlx[vision]
        - Supports Qwen2-VL, Qwen2-Audio, LLaVA, Gemma3-V, etc.
        - Images/audio/video can be URLs, local paths, or data URIs
        - resize_shape in params prevents OOM on large images
        - Streaming uses parent agent's callback handler
    """
    try:
        from strands_mlx.mlx_vision_model import MLXVisionModel
    except ImportError:
        return {
            "status": "error",
            "content": [
                {
                    "text": "Error: mlx-vlm is required for vision models. "
                    "Install with: pip install strands-mlx[vision]"
                }
            ],
        }

    try:
        # Validate at least one media input
        if not any([images, audio, video]):
            return {
                "status": "error",
                "content": [
                    {"text": "Error: At least one of images, audio, or video must be provided"}
                ],
            }

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

        # Create MLX vision model with configuration
        model_config = {}
        if adapter_path:
            model_config["adapter_path"] = adapter_path
        if params:
            model_config["params"] = params

        logger.debug(f"🔄 Creating MLX vision model: {model_id}")
        vision_model = MLXVisionModel(model_id=model_id, **model_config)

        # Build model info string
        media_counts = []
        if images:
            media_counts.append(f"images: {len(images)}")
        if audio:
            media_counts.append(f"audio: {len(audio)}")
        if video:
            media_counts.append(f"video: {len(video)}")

        model_info = f"MLX Vision Model: {model_id}"
        if adapter_path:
            model_info += f" (adapter: {adapter_path})"
        if params:
            model_info += f" | params: {params}"
        model_info += f" | {', '.join(media_counts)}"

        logger.debug(f"--- Model Info ---\n{model_info}")
        logger.debug(f"--- Input Prompt ---\n{prompt}\n")

        # Build prompt with media tags
        prompt_with_media = prompt

        # Add image tags
        if images:
            for img in images:
                prompt_with_media += f" <image>{img}</image>"

        # Add audio tags
        if audio:
            for aud in audio:
                prompt_with_media += f" <audio>{aud}</audio>"

        # Add video tags
        if video:
            for vid in video:
                prompt_with_media += f" <video>{vid}</video>"

        # Create temporary agent with vision model
        logger.debug("🔄 Creating temporary agent with MLX vision model...")
        vision_agent = Agent(
            model=vision_model,
            messages=[],
            tools=filtered_tools,
            system_prompt=system_prompt,
            trace_attributes=trace_attributes,
            **extra_kwargs,
        )

        # Invoke with media-tagged prompt
        result = vision_agent(prompt_with_media)

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

    except FileNotFoundError as e:
        error_msg = f"Error: Media file not found: {str(e)}"
        logger.error(error_msg)
        return {"status": "error", "content": [{"text": error_msg}]}
    except ValueError as e:
        error_msg = f"Error: Invalid input: {str(e)}"
        logger.error(error_msg)
        return {"status": "error", "content": [{"text": error_msg}]}
    except MemoryError as e:
        error_msg = f"Error: Out of memory. Try resize_shape parameter or smaller model: {str(e)}"
        logger.error(error_msg)
        return {"status": "error", "content": [{"text": error_msg}]}
    except Exception as e:
        import traceback

        error_msg = f"Error invoking vision model: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return {"status": "error", "content": [{"text": error_msg}]}
