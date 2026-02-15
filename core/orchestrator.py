"""
Multi-agent orchestrator built on LangGraph.

Reads agent skill.md definitions and builds a LangGraph StateGraph where:
- Each agent is a node with its own system prompt, tools, and model config
- A supervisor agent routes tasks to the right specialist
- Agents can hand off to each other via the handoffs defined in their skill.md
- Shared state flows through the graph

Architecture:
    User → Supervisor → [Researcher | Analyst | Writer | ...] → Supervisor → User

Usage:
    from core.orchestrator import build_multi_agent_graph
    from core.agent_loader import load_all_agents
    from core.chat_model import CitiVertexChat
    from tools.common import TOOL_REGISTRY

    agents = load_all_agents("agents")
    graph = build_multi_agent_graph(agents, TOOL_REGISTRY)
    result = graph.invoke({"messages": [("user", "Analyze Q4 earnings")]})
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent

from core.agent_loader import AgentSkill
from core.chat_model import CitiVertexChat


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

from typing import TypedDict


class AgentState(TypedDict):
    """Shared state that flows through the multi-agent graph."""

    messages: Annotated[list[BaseMessage], operator.add]
    next_agent: str
    context: dict  # shared scratchpad between agents


# ---------------------------------------------------------------------------
# Node builders
# ---------------------------------------------------------------------------

def _make_agent_node(
    skill: AgentSkill,
    tool_registry: dict[str, Any],
):
    """Create a LangGraph node for one agent."""
    # Build LLM with agent-specific config
    llm = CitiVertexChat(
        model_name=skill.model,
        temperature=skill.temperature,
        max_output_tokens=skill.max_tokens,
    )

    # Resolve tool references from registry
    agent_tools = []
    for tool_name in skill.tools:
        if tool_name in tool_registry:
            agent_tools.append(tool_registry[tool_name])
        else:
            raise ValueError(
                f"Agent '{skill.name}' references unknown tool '{tool_name}'. "
                f"Available tools: {list(tool_registry.keys())}"
            )

    # Build the ReAct agent using LangGraph prebuilt
    react_agent = create_react_agent(
        llm,
        tools=agent_tools,
        prompt=skill.system_prompt,
    )

    def node_fn(state: AgentState) -> dict:
        """Run this agent on the current messages."""
        result = react_agent.invoke({"messages": state["messages"]})
        return {"messages": result["messages"]}

    return node_fn


def _make_supervisor_node(
    llm: CitiVertexChat,
    agent_names: list[str],
    agent_descriptions: dict[str, str],
):
    """Create the supervisor node that routes to specialist agents."""
    agents_list = "\n".join(
        f"- **{name}**: {desc}" for name, desc in agent_descriptions.items()
    )

    system_prompt = f"""You are a supervisor coordinating a team of specialist agents.

Available agents:
{agents_list}

Based on the user's request and conversation history, decide which agent
should handle the next step. Respond with ONLY the agent name, or "FINISH"
if the task is complete and you can give the final answer to the user.

Rules:
- Route to the most relevant specialist
- You can route to the same agent multiple times if needed
- Say FINISH only when the user's request is fully addressed
- If an agent's output needs further processing, route to the next agent
"""

    def supervisor_node(state: AgentState) -> dict:
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(messages)
        next_agent = response.content.strip().lower()

        # Normalize — handle variations like "FINISH", "finish", "Finish."
        if "finish" in next_agent:
            next_agent = "FINISH"
        else:
            # Find best match from available agents
            matched = None
            for name in agent_names:
                if name.lower() in next_agent:
                    matched = name
                    break
            next_agent = matched or "FINISH"

        return {"next_agent": next_agent}

    return supervisor_node


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_multi_agent_graph(
    agents: dict[str, AgentSkill],
    tool_registry: dict[str, Any],
    supervisor_model: str = "gemini-2.5-flash",
) -> Any:
    """Build the full multi-agent LangGraph from loaded skill definitions.

    Args:
        agents: Dict of agent name → AgentSkill (from load_all_agents)
        tool_registry: Dict of tool name → LangChain tool instance
        supervisor_model: Model to use for the supervisor

    Returns:
        Compiled LangGraph that can be .invoke()'d
    """
    # Filter out the supervisor if it exists — we build our own
    specialist_agents = {
        name: skill for name, skill in agents.items() if name != "supervisor"
    }

    # Build the graph
    graph = StateGraph(AgentState)

    # Supervisor node
    supervisor_llm = CitiVertexChat(
        model_name=supervisor_model,
        temperature=0.0,
    )
    agent_descriptions = {
        name: skill.description for name, skill in specialist_agents.items()
    }
    supervisor = _make_supervisor_node(
        supervisor_llm,
        list(specialist_agents.keys()),
        agent_descriptions,
    )
    graph.add_node("supervisor", supervisor)

    # Agent nodes
    for name, skill in specialist_agents.items():
        node = _make_agent_node(skill, tool_registry)
        graph.add_node(name, node)
        # After each agent runs, go back to supervisor
        graph.add_edge(name, "supervisor")

    # Supervisor routes to agents or END
    def route_supervisor(state: AgentState) -> str:
        next_agent = state.get("next_agent", "FINISH")
        if next_agent == "FINISH":
            return END
        return next_agent

    graph.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {name: name for name in specialist_agents} | {END: END},
    )

    # Entry point
    graph.set_entry_point("supervisor")

    return graph.compile()


# ---------------------------------------------------------------------------
# Simpler single-agent helper
# ---------------------------------------------------------------------------

def build_single_agent(
    skill: AgentSkill,
    tool_registry: dict[str, Any],
) -> Any:
    """Build a standalone ReAct agent from a single skill definition.

    Useful for testing individual agents before wiring them into
    the multi-agent graph.
    """
    llm = CitiVertexChat(
        model_name=skill.model,
        temperature=skill.temperature,
        max_output_tokens=skill.max_tokens,
    )

    agent_tools = [
        tool_registry[t]
        for t in skill.tools
        if t in tool_registry
    ]

    return create_react_agent(
        llm,
        tools=agent_tools,
        prompt=skill.system_prompt,
    )
