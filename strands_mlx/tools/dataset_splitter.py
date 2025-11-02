"""Dataset splitter for MLX training data.

Splits a single JSONL file into train/valid/test sets for mlx-lm training.
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from strands import tool

logger = logging.getLogger(__name__)


@tool
def dataset_splitter(
    input_path: str,
    output_dir: Optional[str] = None,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """Split JSONL dataset into train/valid/test sets for MLX training.

    Args:
        input_path: Path to input JSONL file (e.g., "session.jsonl")
        output_dir: Output directory (default: same as input file directory)
        train_ratio: Training set ratio (default: 0.8 = 80%)
        valid_ratio: Validation set ratio (default: 0.1 = 10%)
        test_ratio: Test set ratio (default: 0.1 = 10%)
        shuffle: Whether to shuffle data before splitting (default: True)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Dict with split statistics and output paths

    Example:
        dataset_splitter(
            input_path="~/.strands/mlx_training_data/session.jsonl",
            train_ratio=0.8,
            valid_ratio=0.1,
            test_ratio=0.1
        )
    """
    try:
        # Validate ratios
        total_ratio = train_ratio + valid_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.001:
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"Ratios must sum to 1.0 (got {total_ratio:.3f}). "
                        f"Adjust train_ratio={train_ratio}, valid_ratio={valid_ratio}, test_ratio={test_ratio}"
                    }
                ],
            }

        # Expand paths
        input_path = os.path.expanduser(input_path)

        if not os.path.exists(input_path):
            return {
                "status": "error",
                "content": [{"text": f"Input file not found: {input_path}"}],
            }

        # Determine output directory
        if output_dir is None:
            output_dir = os.path.dirname(input_path)
        else:
            output_dir = os.path.expanduser(output_dir)
            os.makedirs(output_dir, exist_ok=True)

        # Read all examples
        examples = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))

        total_examples = len(examples)

        if total_examples == 0:
            return {
                "status": "error",
                "content": [{"text": f"No examples found in {input_path}"}],
            }

        # Shuffle if requested
        if shuffle:
            random.seed(seed)
            random.shuffle(examples)

        # Calculate split sizes
        train_size = int(total_examples * train_ratio)
        valid_size = int(total_examples * valid_ratio)
        test_size = total_examples - train_size - valid_size  # Remaining goes to test

        # Split data
        train_data = examples[:train_size]
        valid_data = examples[train_size : train_size + valid_size]
        test_data = examples[train_size + valid_size :]

        # Write splits
        output_files = {}

        # Write train.jsonl
        train_path = os.path.join(output_dir, "train.jsonl")
        with open(train_path, "w", encoding="utf-8") as f:
            for example in train_data:
                json.dump(example, f, ensure_ascii=False)
                f.write("\n")
        output_files["train"] = train_path

        # Write valid.jsonl
        valid_path = os.path.join(output_dir, "valid.jsonl")
        with open(valid_path, "w", encoding="utf-8") as f:
            for example in valid_data:
                json.dump(example, f, ensure_ascii=False)
                f.write("\n")
        output_files["valid"] = valid_path

        # Write test.jsonl
        test_path = os.path.join(output_dir, "test.jsonl")
        with open(test_path, "w", encoding="utf-8") as f:
            for example in test_data:
                json.dump(example, f, ensure_ascii=False)
                f.write("\n")
        output_files["test"] = test_path

        # Calculate token statistics (approximate)
        def estimate_tokens(text: str) -> int:
            """Rough token estimate: ~4 chars per token."""
            return len(text) // 4

        train_tokens = sum(estimate_tokens(ex["text"]) for ex in train_data)
        valid_tokens = sum(estimate_tokens(ex["text"]) for ex in valid_data)
        test_tokens = sum(estimate_tokens(ex["text"]) for ex in test_data)

        # Build result summary
        result = f"""✅ Dataset split complete!

📊 Split Statistics:
- Total examples: {total_examples}
- Train: {len(train_data)} examples (~{train_tokens:,} tokens)
- Valid: {len(valid_data)} examples (~{valid_tokens:,} tokens)
- Test: {len(test_data)} examples (~{test_tokens:,} tokens)

📁 Output Files:
- Train: {train_path}
- Valid: {valid_path}
- Test: {test_path}

🎯 Ready for training!
Use: mlx_trainer(action="train", data="{output_dir}", ...)
"""

        return {
            "status": "success",
            "content": [
                {"text": result},
                {
                    "json": {
                        "metadata": {
                            "total_examples": total_examples,
                            "train_count": len(train_data),
                            "valid_count": len(valid_data),
                            "test_count": len(test_data),
                            "train_tokens": train_tokens,
                            "valid_tokens": valid_tokens,
                            "test_tokens": test_tokens,
                            "output_dir": output_dir,
                            "files": output_files,
                        },
                    }
                },
            ],
        }

    except Exception as e:
        logger.error(f"Error splitting dataset: {e}", exc_info=True)
        return {
            "status": "error",
            "content": [{"text": f"Error splitting dataset: {str(e)}"}],
        }
