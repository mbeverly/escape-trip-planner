# Escape Trip Planner

An AI-powered tool that searches for escape rooms in a specified area and schedules escape room trips.

## Current Implementation

### Morty Agent (`agents/morty_agent.py`)

A LangGraph agent that searches for escape rooms using the [Morty API](https://mortyapp.com). The agent:

- Uses Claude (claude-sonnet-4) with tool calling to search for escape rooms by region
- Returns structured output as a list of `EscapeRoom` objects with:
  - Company name, room name, description
  - Community score, awards, difficulty
  - Location (lat/long), duration, player count
  - URL

**Graph structure:** `START -> agent -> (tools -> agent)* -> extract -> END`

### Tools (`tools/tools.py`)

- `search_escape_rooms(region)` - Searches the Morty GraphQL API for escape rooms near a given region

## Setup

1. Install dependencies (requires Python 3.13+):
   ```bash
   uv sync
   ```

2. Create a `.env` file with your API key:
   ```
   ANTHROPIC_API_KEY=your_key_here
   ```

3. Run the agent:
   ```bash
   uv run python -m agents.morty_agent
   ```

## Dependencies

- `langchain-anthropic` / `langgraph` - Agent framework
- `pydantic` - Structured output models
- `httpx` - HTTP client for Morty API
- `geopy` - Geocoding regions to coordinates
- `python-dotenv` - Environment variable management

## Planned Features

1. Playwright MCP agent to scrape booking times from suggested rooms
2. Scheduling agent to generate optimized trip schedules (minimizing travel time)

## Notes

- API rate limits and costs should be considered when running the agent
