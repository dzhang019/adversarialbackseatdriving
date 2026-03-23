from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ContrastiveExample:
    id: str
    positive_prompt: str
    negative_prompt: str
    evaluation_prompt: str
    attack_prompt: str
    target_completion: str


def load_examples(path: str | Path) -> list[ContrastiveExample]:
    examples: list[ContrastiveExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            examples.append(
                ContrastiveExample(
                    id=payload.get("id", f"example_{line_number}"),
                    positive_prompt=payload["positive_prompt"],
                    negative_prompt=payload["negative_prompt"],
                    evaluation_prompt=payload["evaluation_prompt"],
                    attack_prompt=payload["attack_prompt"],
                    target_completion=payload["target_completion"],
                )
            )
    if not examples:
        raise ValueError(f"No examples found in {path}")
    return examples
