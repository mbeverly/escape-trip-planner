"""LangGraph agent for searching escape rooms.

This module implements a LangGraph agent that uses the Morty API to search
for escape rooms and returns structured output.
"""

from __future__ import annotations

import os
from typing import Annotated

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from tools.tools import tools

MODEL_NAME = "claude-sonnet-4-20250514"


class EscapeRoom(BaseModel):
    """Structured representation of an escape room."""

    company_name: str = Field(description="The name of the escape room company")
    name: str = Field(description="The name of the escape room")
    description: str = Field(description="A description of the escape room")
    community_score_bucket: str = Field(
        description="The community score bucket (e.g., 'Overwhelmingly Positive')"
    )
    has_awards: bool = Field(description="Whether the escape room has awards")
    is_scary: bool = Field(description="Whether the escape room is scary")
    latitude: float = Field(description="The latitude of the escape room")
    longitude: float = Field(description="The longitude of the escape room")
    minutes: int = Field(description="The duration in minutes")
    players_max: int = Field(description="Maximum number of players")
    players_min: int = Field(description="Minimum number of players")
    difficulty: str = Field(description="The difficulty level")
    url: str = Field(description="The URL of the escape room company")


class EscapeRoomList(BaseModel):
    """Container for a list of escape rooms."""

    escape_rooms: list[EscapeRoom] = Field(description="List of escape rooms")


class AgentState(BaseModel):
    """State container for the LangGraph agent."""

    model_config = {"arbitrary_types_allowed": True}

    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    escape_rooms: list[EscapeRoom] | None = None


tool_node = ToolNode(tools)


def agent_node(state: AgentState) -> dict[str, list[BaseMessage]]:
    """Process the current state and decide on the next action.

    Args:
        state: The current agent state containing messages.

    Returns:
        A dictionary with the new message(s) to add to the state.
    """
    llm = ChatAnthropic(model=MODEL_NAME, temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(state.messages)
    return {"messages": [response]}


def extract_structured_output(state: AgentState) -> dict[str, list[EscapeRoom]]:
    """Extract structured escape room data from the conversation.

    Args:
        state: The current agent state containing messages.

    Returns:
        A dictionary with the extracted escape rooms.
    """
    llm = ChatAnthropic(model=MODEL_NAME, temperature=0)
    llm_structured = llm.with_structured_output(EscapeRoomList)
    response = llm_structured.invoke(state.messages)
    return {"escape_rooms": response.escape_rooms}


def should_continue(state: AgentState) -> str:
    """Determine the next node based on the last message.

    Routes to 'tools' if the last message contains tool calls,
    otherwise routes to 'extract' for structured output extraction.

    Args:
        state: The current agent state containing messages.

    Returns:
        The name of the next node ('tools' or 'extract').
    """
    last_message = state.messages[-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    return "extract"


def create_agent_graph() -> StateGraph:
    """Build and compile the LangGraph agent.

    Graph structure:
        START -> agent -> (tools -> agent)* -> extract -> END

    The agent loops through tools until it has gathered enough information,
    then extracts structured output.

    Returns:
        A compiled LangGraph state graph.
    """
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("extract", extract_structured_output)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, ["tools", "extract"])
    graph.add_edge("tools", "agent")
    graph.add_edge("extract", END)

    return graph.compile()


def main() -> None:
    """Run the escape room search agent."""
    load_dotenv()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY in your .env file")
        return

    agent = create_agent_graph()

    query = "Find highly rated or awarded escape rooms in Toronto CA"
    print(f"Query: {query}\n")
    print("-" * 50)

    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    print(f"\nEscape Rooms:\n{result['escape_rooms']}")


if __name__ == "__main__":
    main()
