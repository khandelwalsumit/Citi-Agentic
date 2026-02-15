"""
Multi-agent orchestrator built on LangGraph.

Reads agent skill.md definitions and builds a LangGraph StateGraph where:
- Each agent is a node with its own system prompt, tools, and model config
- A supervisor agent routes tasks to the right specialist
- Agents can hand off to each other via the handoffs defined in their skill.md
- Shared state flows through the graph
- Full call tree tracing via AgentTracer

Architecture:
    User -> Supervisor -> [Researcher | Analyst | Writer | ...] -> Supervisor -> User
"""

from __future__ import annotations

import logging
import operator
import time
from typing import Annotated, Any, Sequence, TypedDict

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
from core.tracer import AgentTracer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """Shared state that flows through the multi-agent graph."""
    messages: Annotated[list[BaseMessage], operator.add]
    next_agent: str
    context: dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summarize_messages(messages: list[BaseMessage], max_len: int = 300) -> str:
    """Create a short summary of a message list for tracing."""
    parts = []
    for m in messages[-3:]:  # last 3 messages
        role = type(m).__name__.replace("Message", "")
        content = m.content if isinstance(m.content, str) else str(m.content)
        content = content[:100].replace("\n", " ")
        if hasattr(m, "tool_calls") and m.tool_calls:
            tools = ", ".join(tc["name"] for tc in m.tool_calls)
            parts.append(f"{role}: [calls: {tools}]")
        else:
            parts.append(f"{role}: {content}")
    return " | ".join(parts)[:max_len]


# ---------------------------------------------------------------------------
# Node builders
# ---------------------------------------------------------------------------

def _make_agent_node(
    skill: AgentSkill,
    tool_registry: dict[str, Any],
    tracer: AgentTracer | None = None,
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
        react_agent = create_react_agent(llm, tools=agent_tools)
        system_msg = SystemMessage(content=skill.system_prompt)

        def tool_agent_node(state: AgentState) -> dict:
            if tracer:
                tracer.start(skill.name, _summarize_messages(state["messages"]))
            try:
                # Prepend system prompt — CitiVertexChat._generate handles it
                input_msgs = [system_msg] + state["messages"]
                result = react_agent.invoke({"messages": input_msgs})
                # +1 for the system_msg we prepended
                input_len = len(state["messages"]) + 1
                new_messages = result["messages"][input_len:]

                # Log tool calls to tracer
                if tracer:
                    for msg in new_messages:
                        if isinstance(msg, AIMessage) and msg.tool_calls:
                            for tc in msg.tool_calls:
                                tracer.log_tool_call(tc["name"], tc["args"])
                        elif isinstance(msg, ToolMessage):
                            content = msg.content if isinstance(msg.content, str) else str(msg.content)
                            tracer.log_tool_call(
                                msg.name or "tool",
                                {},
                                result=content[:200],
                            )

                if tracer:
                    tracer.end(_summarize_messages(new_messages))
                return {"messages": new_messages}
            except Exception as e:
                if tracer:
                    tracer.end(error=str(e))
                raise

        return tool_agent_node

    else:
        def simple_agent_node(state: AgentState) -> dict:
            if tracer:
                tracer.start(skill.name, _summarize_messages(state["messages"]))
            try:
                messages = [
                    SystemMessage(content=skill.system_prompt),
                    *state["messages"],
                ]
                response = llm.invoke(messages)
                content = response.content if isinstance(response.content, str) else str(response.content)
                if tracer:
                    tracer.end(f"AI: {content[:200]}")
                return {"messages": [response]}
            except Exception as e:
                if tracer:
                    tracer.end(error=str(e))
                raise

        return simple_agent_node


def _make_supervisor_node(
    llm: CitiVertexChat,
    agent_names: list[str],
    agent_descriptions: dict[str, str],
    tracer: AgentTracer | None = None,
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
        if tracer:
            tracer.start("supervisor", _summarize_messages(state["messages"]))

        messages = [
            SystemMessage(content=system_prompt),
            *state["messages"],
        ]
        response = llm.invoke(messages)
        raw = response.content.strip()
        raw_lower = raw.lower().strip().rstrip(".")

        if "finish" in raw_lower:
            next_agent = "FINISH"
        else:
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

        if tracer:
            tracer.log_routing(next_agent)
            tracer.end(f"-> {next_agent}")

        return {"next_agent": next_agent}

    return supervisor_node


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_multi_agent_graph(
    agents: dict[str, AgentSkill],
    tool_registry: dict[str, Any],
    supervisor_model: str = "gemini-2.5-flash",
    tracer: AgentTracer | None = None,
) -> Any:
    """Build the full multi-agent LangGraph from loaded skill definitions.

    Args:
        agents: Dict of agent name -> AgentSkill (from load_all_agents)
        tool_registry: Dict of tool name -> LangChain tool instance
        supervisor_model: Model to use for the supervisor
        tracer: Optional AgentTracer for call tree capture

    Returns:
        Compiled LangGraph that can be .invoke()'d
    """
    specialist_agents = {
        name: skill for name, skill in agents.items() if name != "supervisor"
    }

    if not specialist_agents:
        raise ValueError("No specialist agents found. Add .md files to the agents/ directory.")

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
        tracer=tracer,
    )
    graph.add_node("supervisor", supervisor)

    # Agent nodes
    for name, skill in specialist_agents.items():
        node = _make_agent_node(skill, tool_registry, tracer=tracer)
        graph.add_node(name, node)
        graph.add_edge(name, "supervisor")

    # Routing
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

    graph.set_entry_point("supervisor")
    return graph.compile()


# ---------------------------------------------------------------------------
# Single-agent helper
# ---------------------------------------------------------------------------

def build_single_agent(
    skill: AgentSkill,
    tool_registry: dict[str, Any],
    tracer: AgentTracer | None = None,
) -> Any:
    """Build a standalone agent from a single skill definition."""
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
        react_agent = create_react_agent(llm, tools=agent_tools)
        sys_msg = SystemMessage(content=skill.system_prompt)

        # Wrap to inject system prompt into messages
        from langgraph.graph import MessagesState as _MS

        def call_react(state: _MS):
            result = react_agent.invoke({
                "messages": [sys_msg] + state["messages"],
            })
            # Strip the prepended system msg from output
            return {"messages": result["messages"][len(state["messages"]) + 1:]}

        g = StateGraph(_MS)
        g.add_node("agent", call_react)
        g.set_entry_point("agent")
        g.add_edge("agent", END)
        return g.compile()
    else:
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
