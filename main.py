"""Entry point for the escape trip planner application."""

from __future__ import annotations
import os

from dotenv import load_dotenv
import logging
from datetime import date, timedelta

from agents.escape_room_planner import plan_escape_room_trip


def configure_logging(level: int = logging.INFO) -> None:
    """Configure logging for the application.

    Args:
        level: The logging level (default: INFO)
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Also configure the agents module loggers
    logging.getLogger("agents").setLevel(level)

def main() -> None:
    """Run the escape room planner."""
    # Configure logging first
    configure_logging(logging.INFO)

    load_dotenv()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY in your .env file")
        return

    # Example usage
    region = "Greenville, SC"
    start_date = date.today() + timedelta(days=14)  # 2 weeks from now
    num_days = 4
    group_size = 4
    preferences = "Mix of difficulty levels, prefer mystery and adventure themes, some scary rooms are okay"

    print(f"Planning {num_days}-day escape room trip to {region}")
    print(f"Starting: {start_date}")
    print(f"Group size: {group_size}")
    print(f"Preferences: {preferences}")
    print("=" * 60)
    print()

    itinerary = plan_escape_room_trip(
        region=region,
        start_date=start_date,
        num_days=num_days,
        group_size=group_size,
        preferences=preferences,
    )

    print(itinerary)


if __name__ == "__main__":
    main()
