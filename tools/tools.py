"""Escape room search tools using Morty.

This module provides LangChain-compatible tools for searching escape rooms
that can be used independently for testing or imported by LangGraph agents.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import httpx
from geopy.geocoders import Nominatim
from langchain_core.tools import tool

from util.utils import dump_kb, get_kb_age_seconds, read_kb

# Morty configuration
GRAPHQL_URL = "https://api.mortyapp.com/graphql"
GQL_QUERY_PATH = Path(__file__).parent / "GameFieldsQuery.gql"

# Cache duration in seconds (30 days)
CACHE_DURATION_SECONDS = 30 * 24 * 60 * 60


def _load_graphql_query() -> str:
    """Load the GraphQL query from the .gql file."""
    return GQL_QUERY_PATH.read_text()


def _simplify_game_data(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Transform raw API response into a simplified format.

    Reduces token count by extracting only essential fields from the
    verbose Morty response.

    Args:
        data: Raw JSON response from the Morty GraphQL API.

    Returns:
        List of simplified escape room dictionaries.
    """
    games = data["data"]["games"]["objects"]
    output = []

    for game in games:
        # Extract community score percentages safely
        score_percentages = {p["rating"]: p["percentage"] for p in game["communityScorePercentages"]}

        output.append({
            "company_name": game["gameLocation"]["company"]["name"],
            "url": game["gameLocation"]["company"]["url"],
            "phone_number": game["location"]["phoneNumber"],
            "address": game["location"]["address"]["formatted_address"],
            "name": game["name"],
            "description": game["description"],
            "community_score_bucket": game["communityScoreBucket"],
            "community_rating_count": game["communityRatingCount"],
            "community_score_love": score_percentages.get("LOVE", 0),
            "community_score_like": score_percentages.get("LIKE", 0),
            "community_score_dislike": score_percentages.get("DISLIKE", 0),
            "awards": [
                f"{award['category']['source']['awardName']} - {award['displayTitle']}"
                for award in game["awards"]
            ],
            "latitude": game["location"]["address"]["geometry"]["location"]["lat"],
            "longitude": game["location"]["address"]["geometry"]["location"]["lng"],
            "has_awards": game["hasAwards"],
            "is_scary": game["isScary"],
            "minutes": game["minutes"],
            "min_age": game["minimumAge"],
            "privacy": game["privacy"],
            "players_max": game["playersMax"],
            "players_min": game["playersMin"],
            "difficulty": game["difficulty"],
            "category": game["primaryCategory"]["name"],
        })

    return output


def _fetch_escape_rooms(
    region: str,
    lat: float,
    lng: float,
    page_size: int = 50,
) -> list[dict[str, Any]]:
    """Fetch escape room data from Morty.

    Args:
        region: Name of the region (used for caching).
        lat: Latitude of the search location.
        lng: Longitude of the search location.
        page_size: Maximum number of results to return.

    Returns:
        List of simplified escape room data.

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
    simplified = _simplify_game_data(data)

    # Cache the results
    dump_kb(simplified, f"{region}_escape_rooms.json")

    return simplified


@tool
def search_escape_rooms(region: str) -> list[dict[str, Any]] | str:
    """Search for escape rooms in a specific region using morty.app data.

    Searches Morty for escape rooms near the specified region,
    sorted by community score. Results are cached for 30 days.

    Args:
        region: The region/city to search for escape rooms (e.g., "Boston", "Los Angeles").

    Returns:
        List of escape room data dictionaries, or an error message string.
    """
    try:
        # Geocode the region to coordinates
        geolocator = Nominatim(user_agent="escape-trip-planner")
        location = geolocator.geocode(region)

        if not location:
            return f"Could not find coordinates for '{region}'. Try a more specific location."

        cache_file = f"{region}_escape_rooms.json"

        # Check if we have a recent cached copy
        cache_age = time.time() - get_kb_age_seconds(cache_file)
        if cache_age < CACHE_DURATION_SECONDS:
            return read_kb(cache_file)

        # Fetch fresh data from the API
        return _fetch_escape_rooms(region, location.latitude, location.longitude)

    except httpx.HTTPStatusError as e:
        return f"API error searching for escape rooms: {e}"
    except Exception as e:
        return f"Error searching for escape rooms: {e}"


# List of all tools available to agents
tools = [search_escape_rooms]


if __name__ == "__main__":
    # Allow testing from command line: python -m tools.tools "Boston"
    search_region = sys.argv[1] if len(sys.argv) > 1 else "Boston"
    print(f"Searching for escape rooms in: {search_region}\n")
    print("-" * 50)

    result = search_escape_rooms.invoke(search_region)
    if isinstance(result, list):
        print(f"Found {len(result)} escape rooms")
        for room in result[:5]:
            print(f"  - {room['name']} @ {room['company_name']}")
    else:
        print(result)
