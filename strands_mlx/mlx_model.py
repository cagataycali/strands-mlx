"""MLX model provider for Apple Silicon.

Efficient inference using MLX-LM library for language models on Apple Silicon
with unified memory and Metal acceleration.

- Docs: https://ml-explore.github.io/mlx/
- MLX-LM: https://github.com/ml-explore/mlx-lm
- Models: https://huggingface.co/mlx-community
"""

import json
import logging
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
)

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from pydantic import BaseModel
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
    from huggingface_hub import snapshot_download

    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class MLXModel(Model):
    """MLX model provider.

    Example:
        >>> model = MLXModel(model_id="mlx-community/qwen3-4b-4bit-DWQ")
        >>> model.update_config(params={"temperature": 1, "max_tokens": 3000})
    """

    class MLXConfig(TypedDict, total=False):
        """MLX model configuration."""

        model_id: str
        adapter_path: Optional[str]
        tokenizer_config: Optional[Dict[str, Any]]
        params: Optional[Dict[str, Any]]
        lazy: bool

    def __init__(
        self,
        model_id: str,
        adapter_path: Optional[str] = None,
        **model_config: Unpack[MLXConfig],
    ) -> None:
        """Initialize MLX provider.

        Args:
            model_id: Model identifier (HF Hub ID or local path).
            adapter_path: Optional LoRA adapter path.
            **model_config: Configuration options.
        """
        validate_config_keys(model_config, self.MLXConfig)

        self.config = {
            "model_id": model_id,
            "adapter_path": adapter_path,
            "lazy": False,
            "tokenizer_config": {"trust_remote_code": True},
            **model_config,
        }

        logger.debug("config=<%s> | initializing", self.config)
        self._load_model()

    def _resolve_adapter_path(self, adapter_path: Optional[str]) -> Optional[str]:
        """Resolve adapter path - download from HF if needed.

        Args:
            adapter_path: Local path or HF repo ID.

        Returns:
            Local path to adapter or None.
        """
        if not adapter_path:
            return None

        # Check if it's a local path
        if Path(adapter_path).exists():
            logger.debug("adapter_path=<%s> | local path exists", adapter_path)
            return adapter_path

        # Try to download from HuggingFace
        if HF_HUB_AVAILABLE:
            try:
                logger.debug("adapter_path=<%s> | downloading from HuggingFace", adapter_path)
                local_path = snapshot_download(repo_id=adapter_path, repo_type="model")
                logger.debug("adapter_path=<%s> | downloaded to <%s>", adapter_path, local_path)
                return local_path
            except Exception as e:
                logger.error("adapter_path=<%s> | failed to download: %s", adapter_path, e)
                raise FileNotFoundError(
                    f"Adapter path '{adapter_path}' not found locally and failed to download from HuggingFace: {e}"
                ) from e
        else:
            raise ImportError(
                f"Adapter path '{adapter_path}' not found locally. Install huggingface_hub to download from HuggingFace: pip install huggingface_hub"
            )

    def _load_model(self) -> None:
        """Load model using mlx_lm.load()."""
        model_id = self.config["model_id"]
        logger.debug("model_id=<%s> | loading", model_id)

        # Resolve adapter path (download if needed)
        adapter_path = self._resolve_adapter_path(self.config.get("adapter_path"))

        self.model, self.tokenizer = load(
            model_id,
            tokenizer_config=self.config.get("tokenizer_config", {}),
            adapter_path=adapter_path,
            lazy=self.config.get("lazy", False),
        )

        logger.debug("model loaded")

    @override
    def update_config(self, **model_config: Unpack[MLXConfig]) -> None:  # type: ignore[override]
        """Update configuration."""
        validate_config_keys(model_config, self.MLXConfig)

        if "model_id" in model_config and model_config["model_id"] != self.config.get("model_id"):
            self.config.update(model_config)
            self._load_model()
        else:
            self.config.update(model_config)

    @override
    def get_config(self) -> MLXConfig:
        """Get configuration."""
        return self.config  # type: ignore[return-value]

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
            logger.warning("image content not yet supported in MLX provider")
            return {"type": "text", "text": "[Image content not supported]"}
        if "document" in content:
            logger.warning("document content not yet supported in MLX provider")
            return {"type": "text", "text": "[Document content not supported]"}

        raise TypeError(f"content_type=<{next(iter(content))}> | unsupported type")

    @classmethod
    def format_request_message_tool_call(cls, tool_use: ToolUse) -> Dict[str, Any]:
        """Format a tool call

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
        """Format a tool result message

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

    @classmethod
    def format_request_messages(
        cls, messages: Messages, system_prompt: Optional[str] = None
    ) -> list[Dict[str, Any]]:
        """Format messages array

        Args:
            messages: List of message objects.
            system_prompt: System prompt.

        Returns:
            Formatted messages array.
        """
        formatted_messages: list[Dict[str, Any]]
        formatted_messages = [{"role": "system", "content": system_prompt}] if system_prompt else []

        for message in messages:
            contents = message["content"]

            # Check for reasoningContent and warn user
            if any("reasoningContent" in content for content in contents):
                logger.warning(
                    "reasoningContent in multi-turn conversations requires model support"
                )

            formatted_contents = [
                cls.format_request_message_content(content)
                for content in contents
                if not any(
                    block_type in content
                    for block_type in ["toolResult", "toolUse", "reasoningContent"]
                )
            ]
            formatted_tool_calls = [
                cls.format_request_message_tool_call(content["toolUse"])
                for content in contents
                if "toolUse" in content
            ]
            formatted_tool_messages = [
                cls.format_request_tool_message(content["toolResult"])
                for content in contents
                if "toolResult" in content
            ]

            # Combine text contents into single string
            text_content = " ".join(c["text"] for c in formatted_contents if c["type"] == "text")

            formatted_message = {
                "role": message["role"],
                "content": text_content if text_content else "",
                **({"tool_calls": formatted_tool_calls} if formatted_tool_calls else {}),
            }
            formatted_messages.append(formatted_message)
            formatted_messages.extend(formatted_tool_messages)

        return [
            message
            for message in formatted_messages
            if message["content"] or "tool_calls" in message
        ]

    def format_request(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Format request for mlx-lm

        Args:
            messages: List of message objects.
            tool_specs: List of tool specifications.
            system_prompt: System prompt.

        Returns:
            Formatted request dict.
        """
        formatted_messages = self.format_request_messages(messages, system_prompt)

        # Convert tool specs
        tools = None
        if tool_specs:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool_spec["name"],
                        "description": tool_spec["description"],
                        "parameters": tool_spec["inputSchema"]["json"],
                    },
                }
                for tool_spec in tool_specs
            ]

        return {
            "messages": formatted_messages,
            "tools": tools,
        }

    def format_chunk(self, event: Dict[str, Any]) -> StreamEvent:
        """Format event into StreamEvent.

        Args:
            event: Response event.

        Returns:
            Formatted chunk.
        """
        chunk_type = event["chunk_type"]

        if chunk_type == "message_start":
            return {"messageStart": {"role": "assistant"}}

        if chunk_type == "content_start":
            if event.get("data_type") == "tool":
                return {
                    "contentBlockStart": {
                        "start": {
                            "toolUse": {
                                "name": event["data"]["name"],
                                "toolUseId": event["data"]["id"],
                            }
                        }
                    }
                }
            return {"contentBlockStart": {"start": {}}}

        if chunk_type == "content_delta":
            data_type = event.get("data_type", "text")
            if data_type == "reasoning_content":
                return {
                    "contentBlockDelta": {"delta": {"reasoningContent": {"text": event["data"]}}}
                }
            if data_type == "tool":
                return {"contentBlockDelta": {"delta": {"toolUse": {"input": event["data"]}}}}
            return {"contentBlockDelta": {"delta": {"text": event["data"]}}}

        if chunk_type == "content_stop":
            return {"contentBlockStop": {}}

        if chunk_type == "message_stop":
            reason = event.get("data", "end_turn")
            if reason == "tool_calls":
                return {"messageStop": {"stopReason": "tool_use"}}
            if reason == "length":
                return {"messageStop": {"stopReason": "max_tokens"}}
            return {"messageStop": {"stopReason": "end_turn"}}

        if chunk_type == "metadata":
            return {
                "metadata": {
                    "usage": {
                        "inputTokens": event["data"]["input_tokens"],
                        "outputTokens": event["data"]["output_tokens"],
                        "totalTokens": event["data"]["input_tokens"]
                        + event["data"]["output_tokens"],
                    },
                    "metrics": {"latencyMs": 0},
                },
            }

        raise RuntimeError(f"chunk_type=<{chunk_type}> | unknown type")

    def _stream_switch_content(
        self, data_type: str, prev_data_type: Optional[str]
    ) -> tuple[list[StreamEvent], str]:
        """Handle switching to a new content stream.

        Args:
            data_type: Next content data type.
            prev_data_type: Previous content data type.

        Returns:
            Tuple of (chunks to yield, new data_type).
        """
        chunks = []
        if data_type != prev_data_type:
            if prev_data_type is not None:
                chunks.append(
                    self.format_chunk({"chunk_type": "content_stop", "data_type": prev_data_type})
                )
            chunks.append(
                self.format_chunk({"chunk_type": "content_start", "data_type": data_type})
            )

        return chunks, data_type

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        *,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream conversation.

        Args:
            messages: List of message objects.
            tool_specs: List of tool specifications.
            system_prompt: System prompt.
            tool_choice: Tool choice selection.
            **kwargs: Additional arguments.

        Yields:
            Formatted message chunks.
        """
        warn_on_tool_choice_not_supported(tool_choice)

        logger.debug("formatting request")
        request = self.format_request(messages, tool_specs, system_prompt)
        logger.debug(
            "formatted request=<%s>",
            {**request, "messages": f"{len(request['messages'])} messages"},
        )

        # Apply chat template
        try:
            if request["tools"]:
                prompt = self.tokenizer.apply_chat_template(
                    request["messages"],
                    tools=request["tools"],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            else:
                prompt = self.tokenizer.apply_chat_template(
                    request["messages"],
                    add_generation_prompt=True,
                    tokenize=False,
                )
        except Exception as e:
            logger.warning(f"tools parameter not supported by tokenizer, falling back: {e}")
            # Fallback: add tools to system prompt
            if request["tools"] and request["messages"]:
                tools_desc = "\n\n# Available Tools:\n"
                for tool in request["tools"]:
                    func = tool["function"]
                    tools_desc += f"\n## {func['name']}\n{func['description']}\n"
                    tools_desc += f"Parameters: {json.dumps(func['parameters'], indent=2)}\n"

                if request["messages"][0]["role"] == "system":
                    request["messages"][0]["content"] += tools_desc
                else:
                    request["messages"].insert(0, {"role": "system", "content": tools_desc})

            prompt = self.tokenizer.apply_chat_template(
                request["messages"],
                add_generation_prompt=True,
                tokenize=False,
            )

        # Get params
        params = self.config.get("params", {})
        max_tokens = params.get("max_tokens", 3000)
        temp = params.get("temperature", params.get("temp", 1))
        top_p = params.get("top_p", 1.0)

        sampler = make_sampler(temp=temp, top_p=top_p)

        logger.debug("invoking model")

        # Start streaming
        yield self.format_chunk({"chunk_type": "message_start"})

        # Tracking variables (mlx-lm server.py pattern)
        tool_calls: list[Dict[str, Any]] = []
        tool_text = ""
        in_tool_call = False
        data_type: Optional[str] = None
        token_count = 0
        finish_reason = "end_turn"

        # Generate
        for gen_response in stream_generate(
            self.model, self.tokenizer, prompt, max_tokens=max_tokens, sampler=sampler
        ):
            # Check for tool call markers (mlx-lm server.py pattern lines 678-724)
            if getattr(self.tokenizer, "has_tool_calling", False) and gen_response.text == getattr(
                self.tokenizer, "tool_call_start", None
            ):
                in_tool_call = True
                continue

            if in_tool_call:
                if gen_response.text == getattr(self.tokenizer, "tool_call_end", None):
                    # Parse and store tool call
                    try:
                        tool_data = json.loads(tool_text.strip())
                        tool_calls.append(tool_data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"failed to parse tool call: {e} | text={tool_text}")
                    tool_text = ""
                    in_tool_call = False
                    finish_reason = "tool_calls"
                    continue
                else:
                    tool_text += gen_response.text
                    continue

            # Regular text content
            if gen_response.text:
                chunks, data_type = self._stream_switch_content("text", data_type)
                for chunk in chunks:
                    yield chunk

                yield self.format_chunk(
                    {
                        "chunk_type": "content_delta",
                        "data_type": "text",
                        "data": gen_response.text,
                    }
                )
                token_count += 1

        # Close any open content block
        if data_type:
            yield self.format_chunk({"chunk_type": "content_stop", "data_type": data_type})

        # Emit tool calls
        for tool_call in tool_calls:
            tool_id = f"{tool_call.get('name', 'unknown')}_{token_count}"

            yield self.format_chunk(
                {
                    "chunk_type": "content_start",
                    "data_type": "tool",
                    "data": {"name": tool_call.get("name", "unknown"), "id": tool_id},
                }
            )

            yield self.format_chunk(
                {
                    "chunk_type": "content_delta",
                    "data_type": "tool",
                    "data": json.dumps(tool_call.get("arguments", {})),
                }
            )

            yield self.format_chunk({"chunk_type": "content_stop", "data_type": "tool"})

        # Message stop
        yield self.format_chunk({"chunk_type": "message_stop", "data": finish_reason})

        # Metadata
        input_tokens = len(self.tokenizer.encode(prompt))
        yield self.format_chunk(
            {
                "chunk_type": "metadata",
                "data": {
                    "input_tokens": input_tokens,
                    "output_tokens": token_count,
                },
            }
        )

        logger.debug("finished streaming response from model")

    @override
    async def structured_output(
        self,
        output_model: Type[T],
        prompt: Messages,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Dict[str, Union[T, Any]], None]:
        """Get structured output with JSON schema.

        Args:
            output_model: Pydantic model for output.
            prompt: Prompt messages.
            system_prompt: System prompt.
            **kwargs: Additional arguments.

        Yields:
            Model events with final structured output.
        """
        schema = output_model.model_json_schema()
        json_instruction = f"\n\nRespond with valid JSON:\n{json.dumps(schema, indent=2)}"
        augmented_system_prompt = (system_prompt or "") + json_instruction

        response_text = ""
        async for event in self.stream(prompt, system_prompt=augmented_system_prompt, **kwargs):
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if "text" in delta:
                    response_text += delta["text"]
            yield cast(Dict[str, Union[T, Any]], event)

        try:
            # Extract JSON from markdown if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            data = json.loads(response_text.strip())
            yield {"output": output_model(**data)}
        except Exception as e:
            raise ValueError(
                f"Failed to parse structured output: {e}\nResponse: {response_text}"
            ) from e
