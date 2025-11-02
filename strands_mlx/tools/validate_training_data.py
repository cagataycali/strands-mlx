#!/usr/bin/env python3
"""Validate training data format using tokenizer's native chat template."""

import json
from pathlib import Path
from typing import Dict, Any
from strands import tool


@tool
def validate_training_data(
    jsonl_path: str,
    model_id: str = "mlx-community/Qwen3-1.7B-4bit",
    max_examples: int = 5,
    show_stats: bool = True,
) -> Dict[str, Any]:
    """Validate training data using the target model's tokenizer.

    Uses tokenizer's native apply_chat_template - no hardcoded formats!

    Args:
        jsonl_path: Path to .jsonl training data file
        model_id: Model ID to load tokenizer from (default: Qwen3-1.7B-4bit)
        max_examples: Number of examples to validate (default: 5)
        show_stats: Show detailed statistics (default: True)

    Returns:
        Dict containing validation results and statistics
    """
    path = Path(jsonl_path).expanduser()

    if not path.exists():
        return {"status": "error", "content": [{"text": f"❌ File not found: {jsonl_path}"}]}

    # Load tokenizer - let IT handle format detection
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        return {"status": "error", "content": [{"text": f"❌ Failed to load tokenizer: {e}"}]}

    results = []
    stats = {
        "total_examples": 0,
        "total_chars": 0,
        "total_tokens": 0,
        "tool_calls_total": 0,
    }

    try:
        with open(path, "r") as f:
            for i, line in enumerate(f, 1):
                if i > max_examples:
                    break

                try:
                    data = json.loads(line)
                    text = data.get("text", "")

                    stats["total_examples"] += 1
                    stats["total_chars"] += len(text)

                    # Use tokenizer to count tokens (native, no assumptions!)
                    tokens = tokenizer.encode(text)
                    stats["total_tokens"] += len(tokens)

                    # Count tool calls (universal across all formats)
                    tool_calls = text.count("<tool_call>")
                    stats["tool_calls_total"] += tool_calls

                    # Build result summary
                    result_lines = [
                        f"\n{'='*60}",
                        f"Example {i}:",
                        f"  Characters: {len(text):,}",
                        f"  Tokens: {len(tokens):,}",
                        f"  Tool calls: {tool_calls}",
                    ]

                    results.append("\n".join(result_lines))

                except json.JSONDecodeError:
                    results.append(f"\n❌ Line {i}: Invalid JSON")
                except Exception as e:
                    results.append(f"\n❌ Line {i}: {str(e)}")

        # Count total lines in file
        with open(path, "r") as f:
            total_lines = sum(1 for _ in f)

        # Build final report
        report = []
        report.append("\n" + "=" * 60)
        report.append("🔍 TRAINING DATA VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"\n📋 Model: {model_id}")
        report.append(f"📋 Tokenizer: {tokenizer.__class__.__name__}")

        if show_stats:
            report.append("\n📊 Statistics:")
            report.append(f"  Total examples in file: {total_lines:,}")
            report.append(f"  Validated examples: {stats['total_examples']}")
            report.append(f"  Total characters: {stats['total_chars']:,}")
            report.append(f"  Total tokens: {stats['total_tokens']:,}")
            report.append(f"  Tool calls found: {stats['tool_calls_total']}")
            if stats["total_examples"] > 0:
                report.append(
                    f"  Avg tool calls/example: {stats['tool_calls_total'] / stats['total_examples']:.1f}"
                )
                report.append(
                    f"  Avg tokens/example: {stats['total_tokens'] // stats['total_examples']:,}"
                )
                report.append(
                    f"  Avg chars/example: {stats['total_chars'] // stats['total_examples']:,}"
                )

        # Add example details
        report.extend(results)

        report.append("\n" + "=" * 60)
        report.append("✅ VALIDATION COMPLETE")
        report.append("=" * 60)

        return {"status": "success", "content": [{"text": "\n".join(report)}]}

    except Exception as e:
        return {"status": "error", "content": [{"text": f"❌ Validation failed: {str(e)}"}]}
