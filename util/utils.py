"""Utility functions for the Escape Trip Planner.

This module provides common utilities including:
- Data caching and persistence
- Rate limiting with exponential backoff
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

# Directory for cached data
DATA_DIR = Path("data")

# Directory for saved itineraries
ITINERARIES_DIR = Path("itineraries")

# Rate limiting configuration
MAX_RETRIES = 5
BASE_DELAY = 60.0  # seconds
MAX_DELAY = 300.0  # seconds


def dump_kb(data: Any, name: str = "output.json") -> None:
    """Write data to a JSON file in the data directory.

    Args:
        data: The data to serialize to JSON.
        name: The filename for the output file.
    """
    DATA_DIR.mkdir(exist_ok=True)
    output_path = DATA_DIR / name
    with output_path.open("w") as file:
        json.dump(data, file, indent=2)


def get_kb_age_seconds(name: str) -> float:
    """Get the creation time of a cached file.

    Args:
        name: The filename to check.

    Returns:
        The file's creation time as a Unix timestamp, or 0 if not found.
    """
    try:
        stat = os.stat(DATA_DIR / name)
        return stat.st_birthtime
    except (AttributeError, FileNotFoundError):
        return 0


def read_kb(name: str) -> Any:
    """Read data from a JSON file in the data directory.

    Args:
        name: The filename to read.

    Returns:
        The parsed JSON data.
    """
    path = DATA_DIR / name
    with open(path, "r") as file:
        return json.load(file)


def calculate_backoff_delay(retry_count: int) -> float:
    """Calculate exponential backoff delay with jitter.

    Uses exponential backoff starting from BASE_DELAY, capped at MAX_DELAY,
    with random jitter to avoid thundering herd problems.

    Args:
        retry_count: The current retry attempt number (0-indexed).

    Returns:
        The delay in seconds before the next retry.
    """
    delay = BASE_DELAY * (2**retry_count)
    delay = min(delay, MAX_DELAY)
    # Add jitter (+-25%) to avoid thundering herd
    jitter = delay * 0.25 * (2 * random.random() - 1)
    return delay + jitter


def save_itinerary(itinerary: str, region: str, start_date: str) -> Path:
    """Save an itinerary to a markdown file in the itineraries directory.

    Args:
        itinerary: The itinerary content to save.
        region: The region/city for the trip (used in filename).
        start_date: The start date in YYYY-MM-DD format (used in filename).

    Returns:
        The path to the saved file.
    """
    ITINERARIES_DIR.mkdir(exist_ok=True)

    # Create a safe filename from region and date
    safe_region = region.lower().replace(" ", "_").replace(",", "")
    filename = f"{safe_region}_{start_date}.md"
    filepath = ITINERARIES_DIR / filename

    filepath.write_text(itinerary)
    return filepath
