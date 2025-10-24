import yaml
import os
from typing import Dict, Any


def load_prompt(name: str, version: int = None) -> Dict[str, Any]:
    path = os.path.join(os.path.dirname(__file__), "prompts.yaml")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    candidates = [p for p in data if p.get("name") == name]
    if not candidates:
        raise KeyError(f"Prompt {name} not found")
    if version is None:
        # choose highest version
        return sorted(candidates, key=lambda x: x.get("version", 0))[-1]
    for p in candidates:
        if p.get("version") == version:
            return p
    raise KeyError(f"Prompt {name} v{version} not found")
