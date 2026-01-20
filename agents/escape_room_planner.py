"""LangGraph orchestrator agent for planning escape room trips.

This agent coordinates the local_escape_room_guide and escape_room_reservationist
agents to create a comprehensive 3-4 day escape room itinerary with verified availability.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, timedelta
from typing import Annotated

import httpx

# Configure logging
logger = logging.getLogger(__name__)
from anthropic import RateLimitError
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from util.utils import calculate_backoff_delay, MAX_RETRIES


MODEL_NAME = "claude-sonnet-4-20250514"

SYSTEM_MESSAGE = """You are an expert escape room trip planner who creates exciting, \
well-organized multi-day escape room adventures.

Your role is to orchestrate a complete escape room trip by:

1. GATHERING RECOMMENDATIONS:
   - Use the get_escape_room_recommendations tool to find the best escape rooms in the target region
   - Consider variety in themes (horror, mystery, adventure, sci-fi, etc.)
   - Balance difficulty levels for the group
   - Prioritize highly-rated and award-winning rooms

2. CHECKING AVAILABILITY:
   - Use check_room_availability to verify time slots for your top room choices
   - Only search availability one at a time to not overwhelm the system
   - If a room has no availability, note it and move to alternatives
   - Focus on finding slots that fit a reasonable daily schedule

3. CREATING THE ITINERARY:
   - Plan 2-3 escape rooms per day (allowing ~2-3 hours per room including travel)
   - Schedule rooms at reasonable times (typically 10am-9pm)
   - Group rooms by location/address to minimize travel between venues
   - Include breaks for meals and rest
   - Mix difficulty levels throughout the trip
   - Only include rooms with CONFIRMED availability

4. ITINERARY FORMAT:
   Present the itinerary in an engaging format:

   📅 **Day 1 - [Date]** - [Theme for the day]

   🔐 **[TIME] - [Room Name]** @ [Venue Name]
      - Theme: [Theme/Category]
      - Difficulty: [Easy/Medium/Hard]
      - Duration: [X minutes]
      - Players: [Min-Max]
      - Rating: [Community score]
      - URL: [Venue URL]
      - Address: [Location]
      - ⭐ [Awards if any]

   🍽️ **Lunch/Dinner Break**

   Include at the end:
   - 💰 **Estimated Total Cost**: Based on available pricing
   - 📍 **Key Areas**: Neighborhoods you'll visit
   - 💡 **Pro Tips**: Helpful suggestions

IMPORTANT GUIDELINES:
- Only include rooms with verified availability in the final itinerary
- Be realistic about timing - don't overschedule
- Group rooms geographically to reduce travel time
- Note if a room is scary or has age restrictions
"""


class ScheduledRoom(BaseModel):
    """A room scheduled in the itinerary."""

    day: int = Field(description="Day number (1-4)")
    date: str = Field(description="Date in YYYY-MM-DD format")
    time: str = Field(description="Scheduled time in HH:MM format")
    room_name: str = Field(description="Name of the escape room")
    venue_name: str = Field(description="Name of the escape room company")
    venue_url: str = Field(description="URL of the venue")
    theme: str | None = Field(default=None, description="Room theme/category")
    difficulty: str | None = Field(default=None, description="Difficulty level")
    duration_minutes: int | None = Field(default=None, description="Duration in minutes")
    min_players: int | None = Field(default=None, description="Minimum players")
    max_players: int | None = Field(default=None, description="Maximum players")
    estimated_price: str | None = Field(default=None, description="Estimated price")
    address: str | None = Field(default=None, description="Venue address")
    awards: list[str] = Field(default_factory=list, description="Any awards")
    notes: str | None = Field(default=None, description="Additional notes")


class TripItinerary(BaseModel):
    """Complete escape room trip itinerary."""

    region: str = Field(description="The region/city for the trip")
    start_date: str = Field(description="Trip start date")
    end_date: str = Field(description="Trip end date")
    group_size: int = Field(description="Number of people in the group")
    scheduled_rooms: list[ScheduledRoom] = Field(default_factory=list, description="All scheduled rooms")
    total_estimated_cost: str | None = Field(default=None, description="Total estimated cost")
    summary: str | None = Field(default=None, description="Trip summary/overview")
    pro_tips: list[str] = Field(default_factory=list, description="Helpful tips for the trip")


class AgentState(BaseModel):
    """State container for the orchestrator agent."""

    model_config = {"arbitrary_types_allowed": True}

    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    retry_count: int = 0
    rate_limited: bool = False
    error_message: str | None = None

    # Trip planning state
    region: str | None = None
    start_date: str | None = None
    num_days: int = 4
    group_size: int = 4
    recommendations: list[dict] = Field(default_factory=list)
    availability_results: dict = Field(default_factory=dict)


# Import the sub-agents
from agents.local_escape_room_guide import create_agent_graph as create_guide_graph
from agents.escape_room_reservationist import check_availability_sync


@tool
def get_escape_room_recommendations(region: str, preferences: str = "") -> str:
    """Get escape room recommendations from the local guide agent.

    Args:
        region: The city/region to search for escape rooms (e.g., "Boston", "Los Angeles")
        preferences: Optional preferences like difficulty, themes, group size, etc.

    Returns:
        Recommendations from the escape room guide including room details, ratings, and URLs.
    """
    logger.info(f"[TOOL] Getting escape room recommendations for region: {region}")
    if preferences:
        logger.info(f"[TOOL] Preferences: {preferences}")

    guide_agent = create_guide_graph()

    query = f"Find highly rated or awarded escape rooms in {region}."
    if preferences:
        query += f" Preferences: {preferences}"

    logger.info("[TOOL] Invoking local escape room guide agent...")
    result = guide_agent.invoke({"messages": [HumanMessage(content=query)]})
    final_message = result["messages"][-1]

    content = final_message.content if isinstance(final_message.content, str) else str(final_message.content)
    logger.info(f"[TOOL] Guide agent returned {len(content)} characters of recommendations")

    return content


@tool
def check_room_availability(
    url: str,
    room_name: str,
    target_date: str,
) -> str:
    """Check booking availability for a specific escape room.

    Args:
        url: The URL of the escape room website
        room_name: The name of the specific escape room experience
        target_date: The target date in YYYY-MM-DD format

    Returns:
        Available time slots for the specified date and nearby dates.
    """
    logger.info(f"[TOOL] Checking availability for: {room_name}")
    logger.info(f"[TOOL] URL: {url}")
    logger.info(f"[TOOL] Target date: {target_date}")

    try:
        logger.info("[TOOL] Invoking reservationist agent...")
        result = check_availability_sync(url, room_name, target_date)

        if result.error:
            logger.warning(f"[TOOL] Availability check failed: {result.error}")
            return f"Error checking availability: {result.error}"

        output = f"Availability for {result.escape_room_name} at {result.venue_name}:\n"
        output += f"Target date: {result.target_date}\n"

        if result.available_slots:
            logger.info(f"[TOOL] Found {len(result.available_slots)} time slots")
            output += "\nAvailable slots:\n"
            for slot in result.available_slots:
                status = "✓" if slot.available else "✗"
                output += f"  {status} {slot.date} at {slot.time}"
                if slot.price:
                    output += f" - {slot.price}"
                if slot.spots_remaining:
                    output += f" ({slot.spots_remaining} spots)"
                output += "\n"
        else:
            logger.info("[TOOL] No specific time slots found")
            output += "\nNo specific slots found. "
            if result.booking_notes:
                output += f"Notes: {result.booking_notes}"

        logger.info("[TOOL] Availability check completed")
        return output
    except Exception as e:
        logger.error(f"[TOOL] Exception during availability check: {e}")
        return f"Error checking availability for {room_name}: {e}"


# List of tools available to the orchestrator
tools = [get_escape_room_recommendations, check_room_availability]
tool_node = ToolNode(tools)


def agent_node(state: AgentState) -> dict:
    """Process the current state and decide on the next action."""
    logger.info("[PLANNER] Agent node processing...")
    logger.debug(f"[PLANNER] Current message count: {len(state.messages)}")

    llm = ChatAnthropic(model=MODEL_NAME, temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    messages = [SystemMessage(content=SYSTEM_MESSAGE)] + list(state.messages)

    try:
        logger.info("[PLANNER] Calling LLM...")
        response = llm_with_tools.invoke(messages)

        # Log tool calls if any
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                logger.info(f"[PLANNER] LLM requested tool: {tc['name']}")
        else:
            logger.info("[PLANNER] LLM provided final response (no tool calls)")

        return {"messages": [response], "retry_count": 0}
    except (RateLimitError, httpx.HTTPStatusError) as e:
        if isinstance(e, httpx.HTTPStatusError) and e.response.status_code != 429:
            raise
        new_retry_count = state.retry_count + 1
        logger.warning(f"[PLANNER] Rate limited, retry {new_retry_count}/{MAX_RETRIES}")
        if new_retry_count > MAX_RETRIES:
            logger.error("[PLANNER] Max retries exceeded")
            return {
                "rate_limited": True,
                "error_message": "Service temporarily unavailable due to rate limiting. Please try again later.",
            }
        delay = calculate_backoff_delay(state.retry_count)
        logger.info(f"[PLANNER] Waiting {delay:.1f}s before retry...")
        time.sleep(delay)
        return {"retry_count": new_retry_count}


def handle_rate_limit_error(state: AgentState) -> dict:
    """Return a user-friendly error message when rate limited."""
    error_msg = state.error_message or "Service temporarily unavailable. Please try again later."
    return {"messages": [AIMessage(content=error_msg)]}


def should_continue(state: AgentState) -> str:
    """Determine the next node based on the last message."""
    if state.rate_limited:
        logger.info("[PLANNER] Routing to: error (rate limited)")
        return "error"

    if state.retry_count > 0 and not state.rate_limited:
        logger.info("[PLANNER] Routing to: agent (retry)")
        return "agent"

    last_message = state.messages[-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        logger.info("[PLANNER] Routing to: tools")
        return "tools"

    logger.info("[PLANNER] Routing to: END (conversation complete)")
    return END


def create_planner_graph() -> StateGraph:
    """Build and compile the orchestrator agent.

    Graph structure:
        START -> agent -> (tools -> agent)* -> END
                   |
                   v
                 error -> END
    """
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("error", handle_rate_limit_error)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, ["tools", "error", "agent", END])
    graph.add_edge("tools", "agent")
    graph.add_edge("error", END)

    return graph.compile()


def plan_escape_room_trip(
    region: str,
    start_date: str | date,
    num_days: int = 4,
    group_size: int = 4,
    preferences: str = "",
) -> str:
    """Plan a complete escape room trip.

    Args:
        region: The city/region for the trip (e.g., "Boston", "Los Angeles")
        start_date: The start date of the trip (YYYY-MM-DD string or date object)
        num_days: Number of days for the trip (default: 4)
        group_size: Number of people in the group (default: 4)
        preferences: Optional preferences (themes, difficulty, budget, etc.)

    Returns:
        A complete itinerary for the escape room trip.
    """
    load_dotenv()

    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY not set in environment")
        return "Error: ANTHROPIC_API_KEY not set in environment"

    if isinstance(start_date, date):
        start_date = start_date.isoformat()

    end_date = (date.fromisoformat(start_date) + timedelta(days=num_days - 1)).isoformat()

    logger.info("=" * 60)
    logger.info("[PLANNER] Starting escape room trip planning")
    logger.info(f"[PLANNER] Region: {region}")
    logger.info(f"[PLANNER] Dates: {start_date} to {end_date} ({num_days} days)")
    logger.info(f"[PLANNER] Group size: {group_size}")
    if preferences:
        logger.info(f"[PLANNER] Preferences: {preferences}")
    logger.info("=" * 60)

    planner = create_planner_graph()

    query = f"""Please plan a {num_days}-day escape room adventure in {region}.

Trip Details:
- Region: {region}
- Start Date: {start_date}
- End Date: {end_date}
- Group Size: {group_size} people
- Preferences: {preferences if preferences else "No specific preferences - looking for a variety of highly-rated experiences"}

Please:
1. Use get_escape_room_recommendations to find the best escape rooms in {region}
2. From the recommendations, identify your top 10-15 must-do rooms
3. Use check_room_availability for each top room to find available time slots within the trip dates
4. Create a day-by-day itinerary using rooms with CONFIRMED availability
5. Include all details: times, addresses, themes, difficulty, prices

Make the itinerary engaging and practical!
"""

    logger.info("[PLANNER] Invoking planner agent...")
    result = planner.invoke({
        "messages": [HumanMessage(content=query)],
        "region": region,
        "start_date": start_date,
        "num_days": num_days,
        "group_size": group_size,
    })

    logger.info("[PLANNER] Planning complete!")
    logger.info(f"[PLANNER] Total messages in conversation: {len(result['messages'])}")

    final_message = result["messages"][-1]
    return final_message.content if isinstance(final_message.content, str) else str(final_message.content)


def main() -> None:
    """Run the escape room trip planner and save results."""
    load_dotenv()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY in your .env file")
        return

    region = "Boston"
    start_date = "2026-02-01"
    num_days = 4
    group_size = 4

    print(f"Planning {num_days}-day escape room trip in {region}...")
    print(f"Start date: {start_date}, Group size: {group_size}")
    print("-" * 50)

    itinerary = plan_escape_room_trip(
        region=region,
        start_date=start_date,
        num_days=num_days,
        group_size=group_size,
    )

    # Print to console
    print(itinerary)

    # Save to markdown file in itineraries directory
    from pathlib import Path

    itineraries_dir = Path(__file__).parent.parent / "itineraries"
    itineraries_dir.mkdir(exist_ok=True)

    filename = f"{region.lower().replace(' ', '_')}_{start_date}.md"
    filepath = itineraries_dir / filename

    filepath.write_text(itinerary)
    print("-" * 50)
    print(f"Itinerary saved to: {filepath}")


if __name__ == "__main__":
    main()
