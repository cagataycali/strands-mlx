"""MLX Vision Language Model provider for Apple Silicon.

Vision-enabled inference using MLX-VLM library for multimodal models on Apple Silicon
with unified memory and Metal acceleration.

- MLX-VLM: https://github.com/Blaizzy/mlx-vlm
- Models: https://huggingface.co/mlx-community (Qwen2-VL, LLaVA, Gemma3-V, etc.)
"""

import json
import logging
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Optional,
    Tuple,
    cast,
)

from strands.models._validation import (
    validate_config_keys,
    warn_on_tool_choice_not_supported,
)
from strands.models.model import Model
from strands.types.content import ContentBlock, Messages
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolResult, ToolSpec, ToolUse
from typing_extensions import TypedDict, Unpack, override

try:
    from mlx_vlm import load, stream_generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    from mlx_vlm.video_generate import load_video

    MLX_VLM_AVAILABLE = True
except ImportError:
    MLX_VLM_AVAILABLE = False
    load_video = None

logger = logging.getLogger(__name__)


class MLXVisionModel(Model):
    """MLX Vision Language Model provider.

    Example:
        >>> model = MLXVisionModel(model_id="mlx-community/Qwen2-VL-2B-Instruct-4bit")
        >>> model.update_config(params={"temperature": 0.7, "max_tokens": 2000})
    """

    class MLXVisionConfig(TypedDict, total=False):
        """MLX Vision model configuration."""

        model_id: str
        adapter_path: Optional[str]
        params: Optional[Dict[str, Any]]
        lazy: bool

    def __init__(
        self,
        model_id: str,
        adapter_path: Optional[str] = None,
        **model_config: Unpack[MLXVisionConfig],
    ) -> None:
        """Initialize MLX Vision provider.

        Args:
            model_id: Model identifier (HF Hub ID or local path).
            adapter_path: Optional LoRA adapter path.
            **model_config: Configuration options.
        """
        if not MLX_VLM_AVAILABLE:
            raise ImportError(
                "mlx-vlm is required for vision models. "
                "Install with: pip install strands-mlx[vision]"
            )

        validate_config_keys(model_config, self.MLXVisionConfig)

        self.config = {
            "model_id": model_id,
            "adapter_path": adapter_path,
            "lazy": False,
            **model_config,
        }

        logger.debug("config=<%s> | initializing vision model", self.config)
        self._load_model()

    def _load_model(self) -> None:
        """Load model using mlx_vlm.load()."""
        model_id = self.config["model_id"]
        logger.debug("model_id=<%s> | loading vision model", model_id)

        self.model, self.processor = load(
            model_id,
            adapter_path=self.config.get("adapter_path"),
            trust_remote_code=True,
            lazy=self.config.get("lazy", False),
        )
        self.vlm_config = load_config(model_id, trust_remote_code=True)

        logger.debug("vision model loaded")

    @override
    def update_config(self, **model_config: Unpack[MLXVisionConfig]) -> None:  # type: ignore[override]
        """Update configuration."""
        validate_config_keys(model_config, self.MLXVisionConfig)

        if "model_id" in model_config and model_config["model_id"] != self.config.get("model_id"):
            self.config.update(model_config)
            self._load_model()
        else:
            self.config.update(model_config)

    @override
    def get_config(self) -> MLXVisionConfig:
        """Get configuration."""
        return self.config  # type: ignore[return-value]

    def _extract_media_from_messages(
        self, messages: Messages
    ) -> Tuple[list, Optional[list], Optional[list], Optional[list]]:
        """Extract images, audio, video and format chat messages from Strands messages.

        Args:
            messages: Strands message format

        Returns:
            Tuple of (chat_messages, images_list, audio_list, video_list) where
            chat_messages is a list of dicts with 'role' and 'content' keys,
            ready for apply_chat_template
        """
        import re

        images = []
        audio = []
        video = []
        chat_messages = []

        for message in messages:
            role = message["role"]

            # System messages are handled separately via system_prompt parameter
            if role == "system":
                continue

            # Build content for this message
            text_parts = []
            for content in message["content"]:
                if "image" in content:
                    # Handle Strands image format
                    img_data = content["image"]

                    # Handle Strands SDK format: {format, source: {bytes}}
                    if "format" in img_data and "source" in img_data:
                        source = img_data["source"]
                        if "bytes" in source:
                            # Convert bytes to base64 data URI
                            import base64

                            b64_data = base64.b64encode(source["bytes"]).decode("utf-8")
                            fmt = img_data["format"]
                            images.append(f"data:image/{fmt};base64,{b64_data}")
                        else:
                            logger.warning("Strands SDK image format missing 'bytes' in source")

                    # Handle alternative format: {source: {type, url/data}}
                    elif "source" in img_data:
                        source = img_data["source"]
                        if source["type"] == "url":
                            images.append(source["url"])
                        elif source["type"] == "base64":
                            # mlx_vlm can handle base64 data URIs
                            media_type = source.get("media_type", "image/jpeg")
                            images.append(f"data:{media_type};base64,{source['data']}")

                elif "text" in content:
                    text_content = content["text"]

                    # Extract <image>path</image> tags from text
                    image_pattern = r"<image>(.*?)</image>"
                    found_images = re.findall(image_pattern, text_content)
                    for img_path in found_images:
                        images.append(img_path.strip())
                    text_content = re.sub(image_pattern, "", text_content)

                    # Extract <audio>path</audio> tags from text
                    audio_pattern = r"<audio>(.*?)</audio>"
                    found_audio = re.findall(audio_pattern, text_content)
                    for audio_path in found_audio:
                        audio.append(audio_path.strip())
                    text_content = re.sub(audio_pattern, "", text_content)

                    # Extract <video>path</video> tags from text and load frames
                    video_pattern = r"<video>(.*?)</video>"
                    found_video = re.findall(video_pattern, text_content)
                    for video_path in found_video:
                        video_path = video_path.strip()
                        if load_video is not None:
                            try:
                                # Use mlx-vlm's load_video to extract frames
                                video_data, sample_fps = load_video({"video": video_path})
                                # video_data is np.ndarray (T, C, H, W)
                                video.append(video_data)
                                logger.debug(
                                    f"Extracted video frames: {video_data.shape}, fps={sample_fps}"
                                )
                            except Exception as e:
                                logger.error(f"Failed to load video {video_path}: {e}")
                                # Fallback: treat as regular path
                                video.append(video_path)
                        else:
                            # If load_video not available, just store path
                            video.append(video_path)
                    text_content = re.sub(video_pattern, "", text_content)

                    cleaned_text = text_content.strip()
                    if cleaned_text:
                        text_parts.append(cleaned_text)

            # Create chat message dict (role + content string)
            if text_parts:
                content_text = " ".join(text_parts)
            elif images:
                content_text = "Describe the images."
            elif audio:
                content_text = "Transcribe or describe the audio."
            elif video:
                content_text = "Describe what happens in the video."
            else:
                content_text = "Analyze the media."

            chat_messages.append({"role": role, "content": content_text})

        return (
            chat_messages,
            images if images else None,
            audio if audio else None,
            video if video else None,
        )

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        *,
        tool_choice: Optional[ToolChoice] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream generation with vision, audio, and video support.

        Args:
            messages: Conversation messages with optional images/audio/video
            tool_specs: Available tool specifications (warning: vision models may have limited tool support)
            system_prompt: System prompt
            tool_choice: Tool selection strategy
            **kwargs: Additional arguments (resize_shape, etc.)

        Yields:
            StreamEvent dictionaries with generation progress
        """
        logger.debug("STREAM METHOD CALLED with %d messages", len(messages))
        if tool_choice:
            warn_on_tool_choice_not_supported(tool_choice)

        # Extract media and chat messages from Strands messages
        chat_messages, images, audio, video = self._extract_media_from_messages(messages)

        # Add system message if provided
        if system_prompt:
            chat_messages.insert(0, {"role": "system", "content": system_prompt})

        # Count video frames early (before template application)
        video_frame_count = 0
        if video:
            import numpy as np

            for vid_data in video:
                if isinstance(vid_data, np.ndarray):
                    video_frame_count += vid_data.shape[0]  # T (time/frames) dimension
                else:
                    video_frame_count += 1  # Single fallback

        # Apply chat template with total media counts
        num_images = (len(images) if images else 0) + video_frame_count
        num_audio = len(audio) if audio else 0

        formatted_prompt = apply_chat_template(
            self.processor,
            self.vlm_config,
            chat_messages,  # Pass messages array, not plain text
            num_images=num_images,
            num_audios=num_audio,
        )

        # Get generation parameters
        params = self.config.get("params", {})
        max_tokens = params.get("max_tokens", 2000)
        temperature = params.get("temperature", 0.7)
        top_p = params.get("top_p", 1.0)
        resize_shape = params.get("resize_shape") or kwargs.get("resize_shape")

        logger.debug(
            "generating | prompt_len=%d total_images=%d (orig=%d, video_frames=%d) audio=%d max_tokens=%d temp=%.2f",
            len(formatted_prompt),
            num_images,
            len(images) if images else 0,
            video_frame_count,
            num_audio,
            max_tokens,
            temperature,
        )

        # Yield message start
        yield {"messageStart": {"role": "assistant"}}

        # Yield content block start
        yield {"contentBlockStart": {"start": {}}}

        # Stream generation
        try:
            logger.debug("STREAM: Starting stream_generate")

            # Process video: Convert numpy arrays to PIL Images
            video_images = []
            if video:
                import numpy as np
                from PIL import Image

                for vid_data in video:
                    if isinstance(vid_data, np.ndarray):
                        # video_data is (T, C, H, W) from load_video
                        for frame_idx in range(vid_data.shape[0]):
                            # Extract frame (C, H, W) → (H, W, C) for PIL
                            frame = vid_data[frame_idx].transpose(1, 2, 0)
                            # Convert to uint8 if needed
                            if frame.dtype != np.uint8:
                                frame = (frame * 255).astype(np.uint8)
                            pil_frame = Image.fromarray(frame)
                            video_images.append(pil_frame)
                        logger.debug(
                            f"Converted video to {len(video_images)} PIL frames from shape {vid_data.shape}"
                        )
                    else:
                        # If it's a path (fallback), load as image
                        from mlx_vlm.utils import load_image

                        video_images.append(load_image(vid_data))

            # Combine images + video frames
            combined_images = (images or []) + video_images

            # Build generation kwargs
            gen_kwargs = {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "verbose": False,
            }

            if resize_shape:
                gen_kwargs["resize_shape"] = resize_shape

            for result in stream_generate(
                self.model,
                self.processor,
                formatted_prompt,
                image=combined_images if combined_images else None,
                audio=audio,
                **gen_kwargs,
            ):
                # Yield text delta
                logger.debug(f"STREAM: Got text={repr(result.text[:50])}")
                yield {"contentBlockDelta": {"delta": {"text": result.text}}}

        except Exception as e:
            logger.error("generation error: %s", e)
            raise

        # Yield content block stop
        yield {"contentBlockStop": {}}

        # Yield message delta with stop reason
        yield {"messageDelta": {"stopReason": "end_turn"}}

        # Yield message stop (no content needed)
        yield {}

    @classmethod
    def format_request_message_content(cls, content: ContentBlock) -> Dict[str, Any]:
        """Format a content block.

        Args:
            content: Message content.

        Returns:
            Formatted content dict.
        """
        if "text" in content:
            return {"type": "text", "text": content["text"]}
        if "image" in content:
            img_data = content["image"]
            if "source" in img_data:
                source = img_data["source"]
                if source["type"] == "url":
                    return {"type": "image_url", "image_url": {"url": source["url"]}}
                elif source["type"] == "base64":
                    return {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{source['media_type']};base64,{source['data']}"
                        },
                    }
        if "document" in content:
            logger.warning("document content not yet supported in MLX vision provider")
            return {"type": "text", "text": "[Document content not supported]"}

        raise TypeError(f"content_type=<{next(iter(content))}> | unsupported type")

    @classmethod
    def format_request_message_tool_call(cls, tool_use: ToolUse) -> Dict[str, Any]:
        """Format a tool call.

        Args:
            tool_use: Tool use requested by the model.

        Returns:
            Formatted tool call.
        """
        return {
            "function": {
                "arguments": json.dumps(tool_use["input"]),
                "name": tool_use["name"],
            },
            "id": tool_use["toolUseId"],
            "type": "function",
        }

    @classmethod
    def format_request_tool_message(cls, tool_result: ToolResult) -> Dict[str, Any]:
        """Format a tool result message.

        Args:
            tool_result: Tool result from execution.

        Returns:
            Formatted tool message.
        """
        contents = cast(
            list[ContentBlock],
            [
                {"text": json.dumps(content["json"])} if "json" in content else content
                for content in tool_result["content"]
            ],
        )

        return {
            "role": "tool",
            "tool_call_id": tool_result["toolUseId"],
            "content": [cls.format_request_message_content(content) for content in contents],
        }

    @override
    async def structured_output(
        self,
        output_model: Any,
        prompt: Messages,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Get structured output with JSON schema.

        Note: Structured output support for vision models is limited.
        This method attempts to coerce the model to output JSON but
        vision models may not reliably follow JSON schemas.

        Args:
            output_model: Pydantic model for output.
            prompt: Prompt messages.
            system_prompt: System prompt.
            **kwargs: Additional arguments.

        Yields:
            Model events with final structured output.
        """
        logger.warning("structured_output has limited support for vision models")

        # Get JSON schema
        schema = output_model.model_json_schema()
        json_instruction = f"\n\nRespond with valid JSON:\n{json.dumps(schema, indent=2)}"
        augmented_system_prompt = (system_prompt or "") + json_instruction

        # Stream response
        response_text = ""
        async for event in self.stream(prompt, system_prompt=augmented_system_prompt, **kwargs):
            if event["type"] == "content_block_delta":
                delta = event["delta"]
                if delta["type"] == "text_delta":
                    response_text += delta["text"]
            yield event

        # Try to parse structured output
        try:
            # Extract JSON from markdown if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            data = json.loads(response_text.strip())
            yield {"output": output_model(**data)}
        except Exception as e:
            logger.error(f"Failed to parse structured output: {e}\nResponse: {response_text}")
            raise ValueError(
                f"Failed to parse structured output: {e}\nResponse: {response_text}"
            ) from e
