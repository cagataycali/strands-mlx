"""MLX Session Manager - Model-agnostic training data export.

Uses tokenizer's native chat template for formatting - compatible with mlx-lm training.
Saves conversations turn-by-turn to prevent token overflow.
"""

import json
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from strands.agent.agent import Agent
    from strands.types.content import Message

from strands.session.session_manager import SessionManager

logger = logging.getLogger(__name__)


class MLXSessionManager(SessionManager):
    """MLX-LM compatible session manager - uses tokenizer's chat template.

    Saves conversations turn-by-turn (one user+assistant exchange per line)
    to prevent token overflow during training.
    """

    def __init__(
        self,
        session_id: str,
        tokenizer: Any = None,
        model_id: Optional[str] = None,
        storage_dir: Optional[str] = None,
        add_generation_prompt: bool = False,
        tokenize: bool = False,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize MLX session manager.

        Args:
            session_id: Unique identifier for this session
            tokenizer: Optional tokenizer (will be captured from agent if not provided)
            model_id: Optional model ID to load tokenizer from (e.g., "mlx-community/Qwen3-1.7B-4bit")
            storage_dir: Directory to store training data (default: ~/.strands/mlx_training_data)
            add_generation_prompt: Whether to add generation prompt in template
            tokenize: Whether to tokenize (False for text output)
            system_prompt: Optional system prompt (will be captured from agent if not provided)
            **kwargs: Additional args passed to apply_chat_template
        """
        self.session_id = session_id
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.add_generation_prompt = add_generation_prompt
        self.tokenize = tokenize
        self.system_prompt = system_prompt
        self.template_kwargs = kwargs

        # Storage directory
        self.storage_dir = storage_dir or os.path.expanduser("~/.strands/mlx_training_data")
        os.makedirs(self.storage_dir, exist_ok=True)

        # JSONL file path
        self.jsonl_path = os.path.join(self.storage_dir, f"{session_id}.jsonl")

        # Agent reference (captured during initialize)
        self.agent = None

        # Track last saved message index to save only new turns
        self.last_saved_index = -1

        logger.info(f"MLXSessionManager: session_id={session_id}, output={self.jsonl_path}")

    def _convert_strands_messages_to_chat_format(self, messages: List[Any]) -> List[Dict[str, Any]]:
        """Convert Strands messages to standard chat format.

        Args:
            messages: List of Strands Message objects or dicts

        Returns:
            List of dicts in format: {"role": "...", "content": "..."}
        """
        # DEBUG: Show input messages
        logger.debug(f"\n=== Converting {len(messages)} Strands messages ===")
        for i, msg in enumerate(messages):
            role = msg.get("role") if isinstance(msg, dict) else msg.role
            logger.debug(f"  {i}: role={role}")

        chat_messages = []

        for msg in messages:
            role = msg.get("role") if isinstance(msg, dict) else msg.role
            content = msg.get("content") if isinstance(msg, dict) else msg.content

            if role == "user":
                # Handle user messages and tool results
                message = {"role": "user", "content": ""}

                if isinstance(content, list):
                    text_parts = []

                    for item in content:
                        if isinstance(item, dict):
                            if "text" in item:
                                text_parts.append(item["text"])
                            elif "toolResult" in item:
                                # Tool results become "tool" role messages in ChatML
                                tool_result = item["toolResult"]
                                result_text = []

                                result_content = tool_result.get("content", [])
                                for result_item in result_content:
                                    if isinstance(result_item, dict) and "text" in result_item:
                                        result_text.append(result_item["text"])

                                if result_text:
                                    chat_messages.append(
                                        {
                                            "role": "tool",
                                            "content": "".join(result_text),
                                            "tool_call_id": tool_result.get("toolUseId", ""),
                                        }
                                    )

                    message["content"] = "".join(text_parts)

                # Add user message if has content
                if message["content"]:
                    chat_messages.append(message)

            elif role == "assistant":
                # Handle assistant messages with proper reasoning_content field
                message = {"role": "assistant", "content": ""}
                tool_calls = []

                if isinstance(content, list):
                    reasoning_parts = []
                    text_parts = []

                    for item in content:
                        if isinstance(item, dict):
                            if "reasoningContent" in item:
                                # Handle both text and signature fields
                                # Bedrock can return:
                                #   1. {"reasoningContent": {"reasoningText": {"text": "...", "signature": "..."}}}
                                #   2. {"reasoningContent": {"text": "...", "signature": "..."}}  ← Older format
                                reasoning_content = item["reasoningContent"]

                                # Check for reasoningText wrapper (new Bedrock format)
                                if "reasoningText" in reasoning_content:
                                    reasoning_content = reasoning_content["reasoningText"]

                                # Only use text field (skip cryptographic signatures)
                                if "text" in reasoning_content:
                                    reasoning_text = reasoning_content["text"]
                                    # Only add non-empty reasoning
                                    if reasoning_text.strip():
                                        reasoning_parts.append(reasoning_text)
                            if "text" in item:
                                text_parts.append(item["text"])
                            elif "toolUse" in item:
                                # Extract tool calls
                                tool_use = item["toolUse"]
                                tool_call = {
                                    "id": tool_use.get("toolUseId", ""),
                                    "type": "function",
                                    "function": {
                                        "name": tool_use.get("name"),
                                        "arguments": json.dumps(tool_use.get("input", {})),
                                    },
                                }
                                tool_calls.append(tool_call)

                    # Set reasoning_content field only if non-empty (ChatML will put in <think> tags)
                    if reasoning_parts:
                        combined_reasoning = "".join(reasoning_parts).strip()
                        if combined_reasoning:  # Double check after strip
                            message["reasoning_content"] = combined_reasoning
                            logger.debug(
                                f"  ✅ Set reasoning_content: {combined_reasoning[:50]}..."
                            )

                    # Set content field
                    message["content"] = "".join(text_parts)

                # Add tool calls if present
                if tool_calls:
                    message["tool_calls"] = tool_calls

                # Only add if has content or tool calls
                if message["content"].strip() or tool_calls:
                    chat_messages.append(message)

        # DEBUG: Show converted chat messages
        logger.debug(f"=== Converted to {len(chat_messages)} chat messages ===")
        for i, msg in enumerate(chat_messages):
            role = msg["role"]
            has_tc = bool(msg.get("tool_calls"))
            has_content = bool(msg.get("content", "").strip())
            logger.debug(f"  {i}: role={role}, tool_calls={has_tc}, content={has_content}")

        return chat_messages

    def _extract_tool_specs(self, agent: "Agent") -> Optional[List[Dict[str, Any]]]:
        """Extract tool specifications from agent in OpenAI format.

        Args:
            agent: The Strands agent

        Returns:
            List of tool specs in OpenAI format, or None if no tools
        """
        if not hasattr(agent, "tools") or not agent.tools:
            return None

        tools = []
        for tool in agent.tools:
            try:
                tool_spec = tool.to_strands_tool()
                if isinstance(tool_spec, dict) and "toolSpec" in tool_spec:
                    spec = tool_spec["toolSpec"]
                    # Convert to OpenAI format
                    tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": spec.get("name", "unknown"),
                                "description": spec.get("description", ""),
                                "parameters": spec.get("inputSchema", {}).get("json", {}),
                            },
                        }
                    )
            except Exception as e:
                logger.warning(f"Could not serialize tool: {e}")

        return tools if tools else None

    def _format_turn(
        self,
        system_prompt: str,
        user_messages: List[Dict[str, Any]],
        assistant_messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Format a single turn (user + assistant exchange) using tokenizer's chat template.

        For tool calls, the message order must be:
        - user (question)
        - assistant (tool_calls)
        - tool (results)  ← Between assistant messages!
        - assistant (final response)

        Args:
            system_prompt: System prompt for this turn
            user_messages: List of user/tool messages for this turn
            assistant_messages: List of assistant messages for this turn
            tools: Optional tool specifications

        Returns:
            Formatted text string ready for training
        """
        # DEBUG: Show what we're formatting
        logger.debug("\n_format_turn called:")
        logger.debug(f"  user_messages: {len(user_messages)}")
        for i, msg in enumerate(user_messages):
            logger.debug(f"    {i}: role={msg['role']}")
        logger.debug(f"  assistant_messages: {len(assistant_messages)}")
        for i, msg in enumerate(assistant_messages):
            has_tc = bool(msg.get("tool_calls"))
            logger.debug(f"    {i}: has_tool_calls={has_tc}")

        # Build turn messages: system + user + assistant
        turn_messages = [{"role": "system", "content": system_prompt}]

        # Interleave user/tool messages with assistant messages correctly
        # Pattern 1 (simple): [user, assistant]
        # Pattern 2 (tool): [user, assistant(tool_calls), tool(results), assistant(response)]

        # Separate user messages and tool messages
        pure_user_messages = [msg for msg in user_messages if msg["role"] == "user"]
        tool_messages = [msg for msg in user_messages if msg["role"] == "tool"]

        logger.debug(f"  pure_user: {len(pure_user_messages)}, tool: {len(tool_messages)}")

        # Add initial user message
        if pure_user_messages:
            turn_messages.append(pure_user_messages[0])

        # If we have tool calls, interleave correctly
        if tool_messages and len(assistant_messages) >= 1:
            # Add assistant message with tool_calls
            turn_messages.append(assistant_messages[0])
            # Add tool results
            turn_messages.extend(tool_messages)
            # Add remaining assistant messages (final response)
            turn_messages.extend(assistant_messages[1:])
            logger.debug(f"  → Tool cycle: added {len(tool_messages)} tool messages")
        else:
            # No tool calls, just add all assistant messages
            turn_messages.extend(assistant_messages)
            logger.debug("  → Simple: no tool messages")

        # DEBUG: Show message structure before template
        logger.debug("  Messages being templated:")
        for i, msg in enumerate(turn_messages):
            has_reasoning = "reasoning_content" in msg
            logger.debug(f"    {i}: role={msg['role']}, has_reasoning={has_reasoning}")

        # Use tokenizer's native chat template
        # NOTE: We don't pass tools= parameter because it forces empty <think> tags
        # Tool calls are already formatted in the messages themselves
        try:
            formatted_text = self.tokenizer.apply_chat_template(
                turn_messages,
                tokenize=self.tokenize,
                add_generation_prompt=self.add_generation_prompt,
                **self.template_kwargs,
            )

            # If tokenized, decode back to text
            if self.tokenize:
                formatted_text = self.tokenizer.decode(formatted_text)

            # Strip empty <think> tags (Qwen3 template adds them even when reasoning_content is absent)
            import re

            formatted_text = re.sub(r"<think>\s*</think>\s*", "", formatted_text)

            return formatted_text

        except Exception as e:
            logger.error(f"Error applying chat template: {e}", exc_info=True)
            raise

    def _extract_turns(self, messages: List[Any], start_index: int = 0) -> List[Dict[str, Any]]:
        """Extract complete turns (user + assistant exchanges) from messages.

        A complete turn is:
        - Simple: user -> assistant
        - With tools: user -> assistant(tool_calls) -> tool(results) -> assistant(response)

        Args:
            messages: List of all messages
            start_index: Index to start extracting from

        Returns:
            List of turn dicts: {"user_messages": [...], "assistant_messages": [...]}
        """
        turns = []
        current_turn = {"user_messages": [], "assistant_messages": []}

        # Convert messages starting from start_index
        chat_messages = self._convert_strands_messages_to_chat_format(messages[start_index:])

        # DEBUG: Show converted message sequence
        logger.debug(f"\n=== Converting {len(chat_messages)} messages ===")
        for i, msg in enumerate(chat_messages):
            role = msg["role"]
            has_tool_calls = bool(msg.get("tool_calls"))
            has_content = bool(msg.get("content", "").strip())
            logger.debug(
                f"  {i+1}. role={role}, tool_calls={has_tool_calls}, has_content={has_content}"
            )

        for msg in chat_messages:
            role = msg["role"]

            if role == "user":
                # Start new turn ONLY if previous turn is complete
                # Complete = has final assistant response (not just tool_calls)
                if current_turn["user_messages"] and current_turn["assistant_messages"]:
                    # Check if last assistant has tool_calls
                    last_assistant = current_turn["assistant_messages"][-1]
                    has_pending_tool_calls = last_assistant.get("tool_calls")

                    logger.debug(
                        f"  → Checking if should save turn: has_pending={has_pending_tool_calls}"
                    )

                    # Only save if no pending tool calls (turn is complete)
                    if not has_pending_tool_calls:
                        logger.debug(
                            f"  → Saving turn with {len(current_turn['user_messages'])}U + {len(current_turn['assistant_messages'])}A"
                        )
                        turns.append(current_turn)
                        current_turn = {"user_messages": [], "assistant_messages": []}

                # Add to current turn
                current_turn["user_messages"].append(msg)

            elif role == "tool":
                # Tool results are part of SAME turn (don't start new turn)
                current_turn["user_messages"].append(msg)

            elif role == "assistant":
                # Add to current turn
                current_turn["assistant_messages"].append(msg)

        # Add final turn if complete
        if current_turn["user_messages"] and current_turn["assistant_messages"]:
            has_tool_results = any(
                msg.get("role") == "tool" for msg in current_turn["user_messages"]
            )
            last_assistant = current_turn["assistant_messages"][-1]
            has_tool_calls = last_assistant.get("tool_calls")

            logger.debug("  Final turn check:")
            logger.debug(f"    assistant_messages: {len(current_turn['assistant_messages'])}")
            logger.debug(f"    has_tool_results: {has_tool_results}")
            logger.debug(f"    last_has_tool_calls: {bool(has_tool_calls)}")

            # If last assistant has tool_calls, we MUST have tool_results
            if has_tool_calls and not has_tool_results:
                logger.debug("  → Skipping incomplete turn (tool_calls without results)")
                return turns

            # If we have tool results, we need 2+ assistant messages
            if has_tool_results:
                if len(current_turn["assistant_messages"]) < 2:
                    logger.debug(
                        "  → Skipping incomplete turn (has tool_results but only 1 assistant message)"
                    )
                    return turns

                # Check if last assistant has final response (and no tool_calls)
                has_text_response = last_assistant.get("content", "").strip()

                if has_tool_calls:
                    logger.debug(
                        "  → Skipping incomplete turn (last assistant still has tool_calls)"
                    )
                    return turns

                if not has_text_response:
                    logger.debug("  → Skipping incomplete turn (no final response)")
                    return turns

            # Simple turn (no tools) or complete tool cycle
            logger.debug(
                f"  → Saving final turn with {len(current_turn['user_messages'])}U + {len(current_turn['assistant_messages'])}A"
            )
            turns.append(current_turn)

        logger.debug(f"=== Extracted {len(turns)} turns ===\n")
        return turns

    def initialize(self, agent: "Agent", **kwargs: Any) -> None:
        """Initialize with agent reference.

        Args:
            agent: The Strands agent
            **kwargs: Additional initialization arguments
        """
        self.agent = agent

        # Capture system prompt if not provided
        if not self.system_prompt and hasattr(agent, "system_prompt"):
            self.system_prompt = agent.system_prompt or "You are a helpful AI assistant."

        # Capture tokenizer if not provided
        if self.tokenizer is None:
            # Try loading from model_id first
            if self.model_id:
                try:
                    from transformers import AutoTokenizer

                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                    logger.info(f"Loaded tokenizer from model_id: {self.model_id}")
                except Exception as e:
                    logger.warning(f"Failed to load tokenizer from model_id: {e}")
                    # Continue to try agent.model.tokenizer

            # Fallback to agent's model tokenizer
            if self.tokenizer is None:
                if hasattr(agent, "model") and hasattr(agent.model, "tokenizer"):
                    self.tokenizer = agent.model.tokenizer
                    logger.info("Captured tokenizer from agent.model.tokenizer")
                else:
                    raise ValueError(
                        "No tokenizer available. Provide one of:\n"
                        "  1. tokenizer parameter in __init__\n"
                        "  2. model_id parameter to load tokenizer\n"
                        "  3. Agent with model.tokenizer attribute"
                    )

        # Verify tokenizer has chat template
        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise ValueError(
                "Tokenizer does not have apply_chat_template method. "
                "Use a HuggingFace tokenizer or mlx-lm TokenizerWrapper."
            )

        logger.info(f"Initialized MLX session for agent {agent.agent_id}")

    def append_message(self, message: "Message", agent: "Agent", **kwargs: Any) -> None:
        """Hook called when message is added - we use sync_agent instead."""
        pass

    def sync_agent(self, agent: "Agent", **kwargs: Any) -> None:
        """Save new complete turns (user + assistant exchanges).

        Only saves turns that have a final assistant response (not just tool_calls).

        Args:
            agent: The Strands agent
            **kwargs: Additional sync arguments
        """
        # Nothing to process
        if len(agent.messages) == 0:
            return

        logger.info(f"🔄 sync_agent: {len(agent.messages)} total messages")

        # Get current system prompt
        current_system_prompt = (
            agent.system_prompt
            if hasattr(agent, "system_prompt") and agent.system_prompt
            else self.system_prompt or "You are a helpful AI assistant."
        )

        # Extract tools if available
        tools = self._extract_tool_specs(agent)

        # Extract complete turns from ALL messages (always process from beginning)
        all_turns = self._extract_turns(agent.messages)

        # Count how many COMPLETE turns we've already saved
        # Each saved JSONL line = 1 turn
        already_saved_turn_count = self.get_example_count()

        # Only save NEW complete turns
        new_turns = all_turns[already_saved_turn_count:]

        if not new_turns:
            logger.info(
                f"  → No new complete turns (have {already_saved_turn_count}, found {len(all_turns)})"
            )
            return

        # Save each new turn
        saved_count = 0
        for turn in new_turns:
            try:
                formatted_text = self._format_turn(
                    current_system_prompt, turn["user_messages"], turn["assistant_messages"], tools
                )

                # Save to JSONL
                with open(self.jsonl_path, "a", encoding="utf-8") as f:
                    json.dump({"text": formatted_text}, f, ensure_ascii=False)
                    f.write("\n")

                saved_count += 1
                logger.info(f"  → Saved turn {saved_count}/{len(new_turns)}")

            except Exception as e:
                logger.error(f"Error saving turn: {e}", exc_info=True)

        # Update last_saved_index to the last message we processed
        if saved_count > 0:
            self.last_saved_index = len(agent.messages) - 1
            logger.info(
                f"✅ Saved {saved_count} new turn(s), total: {self.get_example_count()}, last_saved_index: {self.last_saved_index}"
            )

    def redact_latest_message(
        self, redact_message: "Message", agent: "Agent", **kwargs: Any
    ) -> None:
        """Redaction not supported for training data."""
        logger.warning("Redaction not supported for MLX training data export")

    def get_jsonl_path(self) -> str:
        """Get path to JSONL file.

        Returns:
            Path to the JSONL training data file
        """
        return self.jsonl_path

    def get_example_count(self) -> int:
        """Get number of examples saved.

        Returns:
            Number of training examples in the JSONL file
        """
        if not os.path.exists(self.jsonl_path):
            return 0
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MLXSessionManager(session_id='{self.session_id}', "
            f"examples={self.get_example_count()})"
        )
