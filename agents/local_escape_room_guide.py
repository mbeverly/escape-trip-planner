"""LangGraph agent for searching and recommending escape rooms.

This agent represents a local expert on escape rooms in a region.
It uses Morty to search for escape rooms and provides
personalized recommendations based on user preferences.
"""

from __future__ import annotations

import os
import time
from typing import Annotated

import httpx
from anthropic import RateLimitError
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from tools.tools import tools
from util.utils import calculate_backoff_delay, MAX_RETRIES


MODEL_NAME = "claude-sonnet-4-20250514"

SYSTEM_MESSAGE = """You are an enthusiastic local escape room guide with deep knowledge of \
escape room experiences. Your role is to help users find the perfect escape room for their group.

When a user asks about escape rooms in a region, use the search_escape_rooms tool to get \
current data. Then analyze the results and provide personalized recommendations based on:

- Group size
- Experience level and preferred difficulty
- Themes and interests (horror, mystery, adventure, etc.)
- Time constraints and scheduling preferences
- Budget considerations

When making recommendations:
1. Highlight top-rated rooms with strong community scores (look for "Overwhelmingly Positive" or "Very Positive")
2. Note any award-winning experiences
3. Mention if a room is scary or has age restrictions when relevant
4. Include practical details like player count ranges and duration
5. Provide the company name, room name, and URL so users can book
6. Recommend 10-30 rooms to do if possible
7. Always include the name and URL for the escape room in your response
"""


class AgentState(BaseModel):
    """State container for the LangGraph agent."""

    model_config = {"arbitrary_types_allowed": True}

    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    retry_count: int = 0
    rate_limited: bool = False
    error_message: str | None = None


tool_node = ToolNode(tools)


def agent_node(state: AgentState) -> dict:
    """Process the current state and decide on the next action.

    Args:
        state: The current agent state containing messages.

    Returns:
        A dictionary with the new message(s) to add to the state,
        or error state if rate limited.
    """
    llm = ChatAnthropic(model=MODEL_NAME, temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    messages = [SystemMessage(content=SYSTEM_MESSAGE)] + list(state.messages)

    try:
        response = llm_with_tools.invoke(messages)
        return {"messages": [response], "retry_count": 0}
    except (RateLimitError, httpx.HTTPStatusError) as e:
        # Handle both anthropic.RateLimitError and httpx 429 errors
        if isinstance(e, httpx.HTTPStatusError) and e.response.status_code != 429:
            raise
        new_retry_count = state.retry_count + 1
        if new_retry_count > MAX_RETRIES:
            return {
                "rate_limited": True,
                "error_message": "Service temporarily unavailable due to rate limiting. Please try again later.",
            }
        delay = calculate_backoff_delay(state.retry_count)
        time.sleep(delay)
        return {"retry_count": new_retry_count}


def handle_rate_limit_error(state: AgentState) -> dict:
    """Return a user-friendly error message when rate limited.

    Args:
        state: The current agent state.

    Returns:
        A dictionary with an error message in the messages list.
    """
    error_msg = state.error_message or "Service temporarily unavailable. Please try again later."
    return {"messages": [AIMessage(content=error_msg)]}


def should_continue(state: AgentState) -> str:
    """Determine the next node based on the last message.

    Routes to 'error' if rate limited after max retries,
    routes to 'agent' if retrying after rate limit,
    routes to 'tools' if the last message contains tool calls,
    otherwise ends the conversation.

    Args:
        state: The current agent state containing messages.

    Returns:
        The name of the next node ('error', 'agent', 'tools', or END).
    """
    if state.rate_limited:
        return "error"

    # If we just incremented retry_count but aren't rate_limited, retry the agent
    if state.retry_count > 0 and not state.rate_limited:
        return "agent"

    last_message = state.messages[-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    return END


def create_agent_graph() -> StateGraph:
    """Build and compile the LangGraph agent.

    Graph structure:
        START -> agent -> (tools -> agent)* -> END
                   |
                   v
                 error -> END

    The agent loops through tools until it has gathered enough information
    to provide recommendations, then ends. Rate limit errors trigger
    exponential backoff retries, and after max retries routes to error.

    Returns:
        A compiled LangGraph state graph.
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


def main() -> None:
    """Run the escape room search agent."""
    load_dotenv()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY in your .env file")
        return

    agent = create_agent_graph()

    query = "Find highly rated or awarded escape rooms in Boston Massachusetts"
    print(f"Query: {query}\n")
    print("-" * 50)

    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    final_message = result["messages"][-1]
    print(f"\nRecommendations:\n{final_message.content}")


if __name__ == "__main__":
    main()
