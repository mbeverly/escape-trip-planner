"""Escape room search tool using the Morty API.

This module provides tools for searching escape rooms that can be used
independently for testing or imported by the LangGraph agent.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import httpx
from geopy.geocoders import Nominatim
from langchain_core.tools import tool

from util.utils import dump_json

GRAPHQL_URL = "https://api.mortyapp.com/graphql"
GQL_QUERY_PATH = Path(__file__).parent / "GameFieldsQuery.gql"


def _load_graphql_query() -> str:
    """Load the GraphQL query from the .gql file."""
    return GQL_QUERY_PATH.read_text()


def fetch_escape_room_details(
    lat: float,
    lng: float,
    page_size: int = 50,
) -> dict[str, Any]:
    """Fetch escape room details from the Morty API for given coordinates.

    Args:
        lat: Latitude of the search location.
        lng: Longitude of the search location.
        page_size: Maximum number of results to return.

    Returns:
        The JSON response from the Morty GraphQL API.

    Raises:
        httpx.HTTPStatusError: If the API request fails.
    """
    payload = {
        "operationName": "Games",
        "variables": {
            "client": "webapp",
            "distance": 50,
            "filters": {"status": ["COMING_SOON", "OPEN"]},
            "isUser": False,
            "lat": lat,
            "lng": lng,
            "groupByLocation": False,
            "pageSize": page_size,
            "sortBy": "COMMUNITY_SCORE",
        },
        "query": _load_graphql_query(),
    }

    headers = {
        "accept": "*/*",
        "content-type": "application/json",
    }

    response = httpx.post(
        GRAPHQL_URL,
        json=payload,
        headers=headers,
        timeout=30.0,
    )
    response.raise_for_status()

    data = response.json()
    dump_json(data, "response.json")
    return data


@tool
def search_escape_rooms(region: str) -> dict[str, Any] | str:
    """Search for escape rooms in a specific region on morty.app.

    Args:
        region: The region/city to search for escape rooms (e.g., "Boston").

    Returns:
        The escape room data from the API, or an error message string.
    """
    try:
        geolocator = Nominatim(user_agent="escape-trip-planner")
        location = geolocator.geocode(region)

        if not location:
            return f"Could not find coordinates for '{region}'. Try a more specific location."

        print(f"Searching near: {location.address}")
        print(f"Coordinates: {location.latitude}, {location.longitude}")

        return fetch_escape_room_details(location.latitude, location.longitude)

    except httpx.HTTPStatusError as e:
        return f"API error searching for escape rooms: {e}"
    except Exception as e:
        return f"Error searching for escape rooms: {e}"


# List of all tools available to the agent
tools = [search_escape_rooms]


if __name__ == "__main__":
    search_region = sys.argv[1] if len(sys.argv) > 1 else "Boston"
    print(f"Searching for escape rooms in: {search_region}\n")
    print("-" * 50)

    result = search_escape_rooms.invoke(search_region)
    print(result)
    dump_json(result)
