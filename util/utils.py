"""Utility functions for the escape trip planner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DUMP_DIR = Path("dump")


def dump_json(data: Any, name: str = "output.json") -> None:
    """Write data to a JSON file in the dump directory.

    Args:
        data: The data to serialize to JSON.
        name: The filename for the output file.
    """
    DUMP_DIR.mkdir(exist_ok=True)
    output_path = DUMP_DIR / name
    with output_path.open("w") as file:
        json.dump(data, file, indent=2)
