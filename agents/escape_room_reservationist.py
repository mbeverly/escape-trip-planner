"""LangGraph agent for checking escape room booking availability.

This agent uses Playwright via MCP to navigate escape room websites,
find booking pages, and retrieve available time slots for a given date.
"""

from __future__ import annotations

import asyncio
import os
from datetime import date, timedelta
from typing import Annotated, Any

import httpx
from anthropic import RateLimitError
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, Field

from util.utils import calculate_backoff_delay, MAX_RETRIES


MODEL_NAME = "claude-sonnet-4-20250514"

# Only include essential Playwright tools to reduce prompt size
ALLOWED_TOOLS = {
    "browser_navigate",      # Navigate to URLs
    "browser_click",         # Click elements
    "browser_type",          # Type into input fields
    "browser_snapshot",      # Get page content/accessibility tree
    "browser_scroll",        # Scroll the page
    "browser_select_option", # Select from dropdowns (for date pickers)
    "browser_press_key",     # Press keyboard keys
    "browser_go_back",       # Navigate back
    "browser_tab_close",     # Close current tab (use if new tab opens)
    "browser_close",         # IMPORTANT: Close browser when done
}

SYSTEM_MESSAGE = """You are an expert escape room booking assistant specialized in navigating \
escape room websites to find available booking times.

CRITICAL RULES - FOLLOW THESE STRICTLY:

1. STAY ON THE MAIN SITE:
   - NEVER click on social media links (Facebook, Instagram, Twitter, TikTok, etc.)
   - NEVER click on external review sites (Yelp, Google Reviews, TripAdvisor)
   - NEVER click on external payment processors until you've found availability
   - ONLY click links that lead to booking/scheduling pages ON THE SAME DOMAIN
   - If you accidentally navigate away, use browser_go_back IMMEDIATELY

2. TAB MANAGEMENT:
   - Work in a SINGLE TAB only
   - If a new tab opens unexpectedly, close it with browser_tab_close and return to the main tab
   - Check browser_tab_list if you're unsure which tab you're on

3. SELECTIVE CLICKING:
   - Before clicking ANY link, verify it's related to booking/reservations
   - Look for these specific labels ONLY: "Book Now", "Book Online", "Reservations", "Schedule", "Buy Tickets", "Check Availability", "Book This Room"
   - IGNORE: "Learn More", "About Us", "Contact", "FAQ", social icons, footer links (unless clearly booking-related)

NAVIGATION WORKFLOW:

Step 1: INITIAL PAGE LOAD
   - Navigate to the provided URL
   - Take a snapshot to understand the page structure
   - Identify the booking/reservation button or link (usually prominent on the page)

Step 2: FIND BOOKING PAGE
   - Click ONLY the booking-related link
   - If the booking is embedded on the homepage, look for a calendar or date picker
   - Common booking systems: FareHarbor, Xola, Bookeo, Peek, Checkfront - these often load in iframes

Step 3: SELECT THE EXPERIENCE
   - Find the specific escape room by name (may be listed as cards or dropdown)
   - The name might be slightly different - match the closest one
   - Click to select it

Step 4: CHECK DATES
   - Find the calendar/date picker
   - Navigate to the target date
   - If unavailable, check the next 3-5 days
   - Look for visual indicators (green = available, gray/red = unavailable)

Step 5: EXTRACT TIMES
   - Once a date is selected, list all available time slots
   - Note prices if visible
   - Note group size limits if shown

Step 6: REPORT RESULTS
   - List available dates and times clearly
   - Include any relevant booking constraints
   - If no availability found, say so clearly

TROUBLESHOOTING:
- If the page seems stuck, try browser_snapshot to see current state
- If you navigated to wrong page, use browser_go_back
- If booking is in an iframe, the snapshot should still capture it
- If you hit a CAPTCHA or login wall, report it and stop

CRITICAL - CLEANUP:
- When you have found the availability information (or determined it's not available), you are DONE
- ALWAYS call browser_close as your FINAL action before responding
- This ensures the browser is properly closed and resources are freed

BE EFFICIENT: Get in, find availability, close browser, report results.
"""


class TimeSlot(BaseModel):
    """A single available booking time slot."""

    date: str = Field(description="The date in YYYY-MM-DD format")
    time: str = Field(description="The time in HH:MM format (24-hour)")
    available: bool = Field(description="Whether this slot is available")
    spots_remaining: int | None = Field(default=None, description="Number of spots remaining if known")
    price: str | None = Field(default=None, description="Price for this time slot if displayed")
    notes: str | None = Field(default=None, description="Any additional notes about this slot")


class BookingAvailability(BaseModel):
    """Structured output for escape room booking availability."""

    escape_room_name: str = Field(description="Name of the escape room experience")
    venue_name: str = Field(description="Name of the escape room company/venue")
    url: str = Field(description="URL of the booking page")
    target_date: str = Field(description="The originally requested date")
    available_slots: list[TimeSlot] = Field(default_factory=list, description="List of available time slots")
    dates_checked: list[str] = Field(default_factory=list, description="All dates that were checked")
    booking_notes: str | None = Field(default=None, description="Any general notes about booking")
    error: str | None = Field(default=None, description="Error message if booking info couldn't be retrieved")


class AgentState(BaseModel):
    """State container for the LangGraph agent."""

    model_config = {"arbitrary_types_allowed": True}

    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    retry_count: int = 0
    rate_limited: bool = False
    error_message: str | None = None
    mcp_session: Any | None = Field(default=None, exclude=True)


def create_agent_node(mcp_tools: list):
    """Create an agent node with access to MCP tools."""

    async def agent_node(state: AgentState) -> dict:
        """Process the current state and decide on the next action."""
        llm = ChatAnthropic(model=MODEL_NAME, temperature=0)

        # Convert MCP tools to LangChain tool format, filtering to essential tools only
        tools_for_llm = []
        for mcp_tool in mcp_tools:
            # Only include whitelisted tools to reduce prompt size
            if mcp_tool.name not in ALLOWED_TOOLS:
                continue
            tool_schema = {
                "name": mcp_tool.name,
                "description": mcp_tool.description or f"Playwright tool: {mcp_tool.name}",
                "input_schema": mcp_tool.inputSchema if hasattr(mcp_tool, 'inputSchema') else {"type": "object", "properties": {}},
            }
            tools_for_llm.append(tool_schema)

        llm_with_tools = llm.bind_tools(tools_for_llm)
        messages = [SystemMessage(content=SYSTEM_MESSAGE)] + list(state.messages)

        try:
            response = await llm_with_tools.ainvoke(messages)
            return {"messages": [response], "retry_count": 0}
        except (RateLimitError, httpx.HTTPStatusError) as e:
            if isinstance(e, httpx.HTTPStatusError) and e.response.status_code != 429:
                raise
            new_retry_count = state.retry_count + 1
            if new_retry_count > MAX_RETRIES:
                return {
                    "rate_limited": True,
                    "error_message": "Service temporarily unavailable due to rate limiting. Please try again later.",
                }
            delay = calculate_backoff_delay(state.retry_count)
            await asyncio.sleep(delay)
            return {"retry_count": new_retry_count}

    return agent_node


def create_tool_node(mcp_session: ClientSession):
    """Create a tool node that executes MCP tools."""
    from langchain_core.messages import ToolMessage

    async def tool_node(state: AgentState) -> dict:
        """Execute tools called by the agent."""
        last_message = state.messages[-1]

        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {"messages": []}

        tool_messages = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            try:
                result = await mcp_session.call_tool(tool_name, tool_args)
                # Extract content from the result
                if hasattr(result, 'content') and result.content:
                    content = "\n".join(
                        item.text if hasattr(item, 'text') else str(item)
                        for item in result.content
                    )
                else:
                    content = str(result)
            except Exception as e:
                content = f"Error executing {tool_name}: {e}"

            tool_messages.append(
                ToolMessage(content=content, tool_call_id=tool_call["id"])
            )

        return {"messages": tool_messages}

    return tool_node


def handle_rate_limit_error(state: AgentState) -> dict:
    """Return a user-friendly error message when rate limited."""
    error_msg = state.error_message or "Service temporarily unavailable. Please try again later."
    return {"messages": [AIMessage(content=error_msg)]}


def should_continue(state: AgentState) -> str:
    """Determine the next node based on the last message."""
    if state.rate_limited:
        return "error"

    if state.retry_count > 0 and not state.rate_limited:
        return "agent"

    last_message = state.messages[-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    return END


def create_agent_graph(mcp_session: ClientSession, mcp_tools: list) -> StateGraph:
    """Build and compile the LangGraph agent with MCP tools.

    Graph structure:
        START -> agent -> (tools -> agent)* -> END
                   |
                   v
                 error -> END
    """
    graph = StateGraph(AgentState)

    agent_node = create_agent_node(mcp_tools)
    tool_node = create_tool_node(mcp_session)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("error", handle_rate_limit_error)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, ["tools", "error", "agent", END])
    graph.add_edge("tools", "agent")
    graph.add_edge("error", END)

    return graph.compile()


async def check_availability(
    url: str,
    experience_name: str,
    target_date: str | date,
) -> BookingAvailability:
    """Check booking availability for an escape room.

    Args:
        url: The URL of the escape room website.
        experience_name: The name of the specific escape room experience.
        target_date: The target date for booking (YYYY-MM-DD string or date object).

    Returns:
        BookingAvailability with the available time slots.
    """
    load_dotenv()

    if not os.getenv("ANTHROPIC_API_KEY"):
        return BookingAvailability(
            escape_room_name=experience_name,
            venue_name="Unknown",
            url=url,
            target_date=str(target_date),
            error="ANTHROPIC_API_KEY not set in environment",
        )

    # Convert date to string if needed
    if isinstance(target_date, date):
        target_date = target_date.isoformat()

    server_params = StdioServerParameters(
        command="npx",
        args=["@playwright/mcp@latest", "--headless"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()

            # Get available tools
            tools_result = await session.list_tools()
            mcp_tools = tools_result.tools

            # Create and run the agent
            agent = create_agent_graph(session, mcp_tools)

            query = f"""Please find available booking times for the escape room experience.

Escape Room Details:
- Website URL: {url}
- Experience Name: {experience_name}
- Target Date: {target_date}

Navigate to the website, find the booking page for "{experience_name}", and retrieve all available
time slots for {target_date}. If that date has no availability, check up to 3 days after.

Return the results as structured data including:
- All available time slots with dates and times
- Any pricing information visible
- Any constraints (group size, etc.)
- Which dates you checked
"""

            # Run the agent
            result = await agent.ainvoke({"messages": [HumanMessage(content=query)]})

            final_message = result["messages"][-1]

            # Parse the agent's response into structured output
            # The agent should provide structured availability info
            return BookingAvailability(
                escape_room_name=experience_name,
                venue_name="Extracted from agent response",
                url=url,
                target_date=target_date,
                booking_notes=final_message.content if isinstance(final_message.content, str) else str(final_message.content),
            )


def check_availability_sync(
    url: str,
    experience_name: str,
    target_date: str | date,
) -> BookingAvailability:
    """Synchronous wrapper for check_availability.

    Args:
        url: The URL of the escape room website.
        experience_name: The name of the specific escape room experience.
        target_date: The target date for booking (YYYY-MM-DD string or date object).

    Returns:
        BookingAvailability with the available time slots.
    """
    return asyncio.run(check_availability(url, experience_name, target_date))


async def main() -> None:
    """Run the escape room reservationist agent."""
    load_dotenv()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY in your .env file")
        return

    # Example usage
    url = "http://redfoxescapes.com"
    experience_name = "The Body Shop"
    target_date = date.today() + timedelta(days=7)

    print(f"Checking availability for: {experience_name}")
    print(f"URL: {url}")
    print(f"Target date: {target_date}")
    print("-" * 50)

    result = await check_availability(url, experience_name, target_date)

    print(f"\nResults:")
    print(f"Experience: {result.escape_room_name}")
    print(f"Target Date: {result.target_date}")
    print(f"Dates Checked: {result.dates_checked}")
    print(f"\nAvailable Slots:")
    for slot in result.available_slots:
        status = "✓ Available" if slot.available else "✗ Unavailable"
        print(f"  {slot.date} {slot.time} - {status}")
        if slot.price:
            print(f"    Price: {slot.price}")
        if slot.spots_remaining:
            print(f"    Spots: {slot.spots_remaining}")

    if result.booking_notes:
        print(f"\nNotes: {result.booking_notes}")

    if result.error:
        print(f"\nError: {result.error}")


if __name__ == "__main__":
    asyncio.run(main())
