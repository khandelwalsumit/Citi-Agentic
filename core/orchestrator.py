"""
Multi-agent orchestrator built on LangGraph.

Reads agent skill.md definitions and builds a LangGraph StateGraph where:
- Each agent is a node with its own system prompt, tools, and model config
- A supervisor agent routes tasks to the right specialist
- Agents can hand off to each other via the handoffs defined in their skill.md
- Shared state flows through the graph

Architecture:
    User -> Supervisor -> [Researcher | Analyst | Writer | ...] -> Supervisor -> User

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

import logging
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

logger = logging.getLogger(__name__)


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
    """Create a LangGraph node for one agent.

    For agents WITH tools  -> uses create_react_agent (ReAct loop).
    For agents WITHOUT tools -> direct LLM call with system prompt.
    """
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
            logger.warning(
                "Agent '%s' references unknown tool '%s'. Skipping. "
                "Available: %s",
                skill.name, tool_name, list(tool_registry.keys()),
            )

    if agent_tools:
        # ---- Agent with tools: use ReAct agent ----
        # create_react_agent in langgraph-prebuilt 1.0.x uses
        # `state_modifier` to inject the system prompt.
        react_agent = create_react_agent(
            llm,
            tools=agent_tools,
            state_modifier=skill.system_prompt,
        )

        def tool_agent_node(state: AgentState) -> dict:
            result = react_agent.invoke({"messages": state["messages"]})
            # Only return NEW messages (not the full history) to avoid duplication
            input_len = len(state["messages"])
            new_messages = result["messages"][input_len:]
            return {"messages": new_messages}

        return tool_agent_node

    else:
        # ---- Agent without tools: direct LLM call ----
        def simple_agent_node(state: AgentState) -> dict:
            messages = [
                SystemMessage(content=skill.system_prompt),
                *state["messages"],
            ]
            response = llm.invoke(messages)
            return {"messages": [response]}

        return simple_agent_node


def _make_supervisor_node(
    llm: CitiVertexChat,
    agent_names: list[str],
    agent_descriptions: dict[str, str],
):
    """Create the supervisor node that routes to specialist agents."""
    agents_list = "\n".join(
        f"- {name}: {desc}" for name, desc in agent_descriptions.items()
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
- Respond with a single word: one of [{', '.join(agent_names)}, FINISH]"""

    def supervisor_node(state: AgentState) -> dict:
        messages = [
            SystemMessage(content=system_prompt),
            *state["messages"],
        ]
        response = llm.invoke(messages)
        raw = response.content.strip()

        # Normalize — handle variations like "FINISH", "finish", "Finish."
        raw_lower = raw.lower().strip().rstrip(".")

        if "finish" in raw_lower:
            next_agent = "FINISH"
        else:
            # Find best match from available agents
            matched = None
            for name in agent_names:
                if name.lower() in raw_lower:
                    matched = name
                    break
            if not matched:
                logger.warning(
                    "Supervisor returned unrecognized agent: '%s'. "
                    "Expected one of %s. Defaulting to FINISH.",
                    raw, agent_names,
                )
            next_agent = matched or "FINISH"

        logger.info("Supervisor routed to: %s (raw: '%s')", next_agent, raw)
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
        agents: Dict of agent name -> AgentSkill (from load_all_agents)
        tool_registry: Dict of tool name -> LangChain tool instance
        supervisor_model: Model to use for the supervisor

    Returns:
        Compiled LangGraph that can be .invoke()'d
    """
    # Filter out the supervisor if it exists — we build our own
    specialist_agents = {
        name: skill for name, skill in agents.items() if name != "supervisor"
    }

    if not specialist_agents:
        raise ValueError("No specialist agents found. Add .md files to the agents/ directory.")

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

    if agent_tools:
        return create_react_agent(
            llm,
            tools=agent_tools,
            state_modifier=skill.system_prompt,
        )
    else:
        # For agents without tools, wrap in a simple graph
        from langgraph.graph import MessagesState

        def call_llm(state: MessagesState):
            messages = [
                SystemMessage(content=skill.system_prompt),
                *state["messages"],
            ]
            response = llm.invoke(messages)
            return {"messages": [response]}

        graph = StateGraph(MessagesState)
        graph.add_node("llm", call_llm)
        graph.set_entry_point("llm")
        graph.add_edge("llm", END)
        return graph.compile()
