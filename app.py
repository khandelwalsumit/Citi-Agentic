"""
Streamlit UI for Citi Agentic Multi-Agent System.

Features:
- Chat interface for interacting with the multi-agent system
- Live call tree showing agent routing, tool calls, and timing
- Agent selector for single-agent testing
- Debug panel with full message traces

Run:
    streamlit run app.py
"""

import os
import sys
import time
import streamlit as st

# ---- Page config (must be first Streamlit call) ----
st.set_page_config(
    page_title="Citi Agentic",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

import config
from core.auth import get_api_key, init_vertex
from core.chat_model import CitiVertexChat
from core.agent_loader import load_all_agents
from core.orchestrator import build_multi_agent_graph, build_single_agent
from core.tracer import AgentTracer, AgentTrace
from tools.common import TOOL_REGISTRY


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Tighter spacing */
    .block-container { padding-top: 1rem; }

    /* Call tree styling */
    .trace-node {
        border-left: 3px solid #0066cc;
        padding: 6px 12px;
        margin: 4px 0 4px 8px;
        font-size: 0.85rem;
        background: #f8f9fa;
        border-radius: 0 4px 4px 0;
    }
    .trace-node.supervisor { border-left-color: #6c757d; }
    .trace-node.error { border-left-color: #dc3545; background: #fff5f5; }

    .trace-tool {
        font-size: 0.8rem;
        color: #495057;
        padding: 2px 0 2px 16px;
    }

    .trace-routing {
        font-size: 0.8rem;
        font-weight: 600;
        color: #0066cc;
        padding: 2px 0 2px 16px;
    }

    .trace-timing {
        font-size: 0.75rem;
        color: #6c757d;
    }

    .agent-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 4px;
    }
    .agent-badge.supervisor { background: #e9ecef; color: #495057; }
    .agent-badge.researcher { background: #cce5ff; color: #004085; }
    .agent-badge.analyst { background: #d4edda; color: #155724; }
    .agent-badge.writer { background: #fff3cd; color: #856404; }
    .agent-badge.error { background: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "traces" not in st.session_state:
    st.session_state.traces = []  # list of (query, AgentTracer) pairs
if "agents" not in st.session_state:
    st.session_state.agents = {}


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def authenticate():
    """Initialize Vertex AI auth from .env or sidebar inputs."""
    os.environ["REQUESTS_CA_BUNDLE"] = config.CA_BUNDLE

    if config.COIN_CLIENT_ID:
        try:
            token = get_api_key(
                config.COIN_CLIENT_ID,
                config.COIN_CLIENT_SECRET,
                config.COIN_CLIENT_SCOPES,
            )
            init_vertex(
                token=token,
                project=config.R2D2_PROJECT,
                api_endpoint=config.R2D2_ENDPOINT,
                username=config.R2D2_USERNAME,
            )
            st.session_state.authenticated = True
            return True
        except Exception as e:
            st.error(f"Authentication failed: {e}")
            return False
    return False


# ---------------------------------------------------------------------------
# Call tree renderer
# ---------------------------------------------------------------------------

def render_trace_node(node: dict, depth: int = 0):
    """Render a single trace node and its children recursively."""
    agent = node["agent"]
    duration = node["duration_ms"]
    status = node["status"]

    # Badge color
    badge_class = agent if agent in ("supervisor", "researcher", "analyst", "writer") else "supervisor"
    if status == "error":
        badge_class = "error"

    # Status icon
    status_icon = {"done": "✓", "running": "⟳", "error": "✗"}.get(status, "?")

    # Node header
    indent = "&nbsp;" * (depth * 4)
    st.markdown(
        f'<div class="trace-node {"supervisor" if agent == "supervisor" else ""} '
        f'{"error" if status == "error" else ""}" '
        f'style="margin-left: {depth * 20}px;">'
        f'<span class="agent-badge {badge_class}">{agent}</span> '
        f'{status_icon} '
        f'<span class="trace-timing">{duration:.0f}ms</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Routing decision (for supervisor)
    if node.get("routing"):
        st.markdown(
            f'<div class="trace-routing" style="margin-left: {depth * 20 + 20}px;">'
            f'→ {node["routing"]}</div>',
            unsafe_allow_html=True,
        )

    # Tool calls
    for tc in node.get("tool_calls", []):
        args_str = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items()) if tc["args"] else ""
        result_preview = tc.get("result", "")[:80]
        st.markdown(
            f'<div class="trace-tool" style="margin-left: {depth * 20 + 20}px;">'
            f'🔧 <code>{tc["name"]}({args_str})</code>'
            f'{f" → {result_preview}" if result_preview else ""}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Error
    if node.get("error"):
        st.markdown(
            f'<div class="trace-tool" style="margin-left: {depth * 20 + 20}px; color: #dc3545;">'
            f'Error: {node["error"][:200]}</div>',
            unsafe_allow_html=True,
        )

    # Children
    for child in node.get("children", []):
        render_trace_node(child, depth + 1)


def render_trace_summary(tracer: AgentTracer):
    """Render summary stats for a trace."""
    col1, col2, col3 = st.columns(3)
    col1.metric("Agents Called", tracer.agent_count)
    col2.metric("Tool Calls", tracer.tool_call_count)
    col3.metric("Total Time", f"{tracer.total_duration_ms:.0f}ms")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Citi Agentic")
    st.caption("Multi-Agent System")

    # Auth status
    if not st.session_state.authenticated:
        st.warning("Not authenticated")
        if st.button("Connect to R2D2"):
            with st.spinner("Authenticating..."):
                if authenticate():
                    st.success("Connected!")
                    st.rerun()
    else:
        st.success("Connected to R2D2")

    st.divider()

    # Load agents
    try:
        agents = load_all_agents(config.AGENTS_DIR)
        st.session_state.agents = agents
    except FileNotFoundError:
        agents = {}
        st.warning(f"No agents directory at: {config.AGENTS_DIR}")

    # Agent listing
    st.subheader("Agents")
    for name, skill in agents.items():
        with st.expander(f"📋 {name}", expanded=False):
            st.write(f"**Model:** {skill.model}")
            st.write(f"**Temp:** {skill.temperature}")
            st.write(f"**Tools:** {', '.join(skill.tools) if skill.tools else 'none'}")
            st.write(f"**Handoffs:** {', '.join(skill.handoffs) if skill.handoffs else 'none'}")
            st.text_area(
                "System Prompt",
                skill.system_prompt[:500],
                height=100,
                disabled=True,
                key=f"prompt_{name}",
            )

    st.divider()

    # Mode selector
    mode = st.radio(
        "Mode",
        ["Multi-Agent", "Single Agent"],
        help="Multi-Agent uses supervisor routing. Single Agent runs one agent directly.",
    )

    selected_agent = None
    if mode == "Single Agent":
        agent_names = list(agents.keys())
        if agent_names:
            selected_agent = st.selectbox("Select Agent", agent_names)

    st.divider()

    # Clear chat
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.traces = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main layout: Chat + Debug panel
# ---------------------------------------------------------------------------

chat_col, debug_col = st.columns([3, 2])

# ---- Chat panel ----
with chat_col:
    st.header("Chat")

    # Display message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input(
        "Ask the agents..." if st.session_state.authenticated else "Connect to R2D2 first",
        disabled=not st.session_state.authenticated,
    ):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Run agents
        with st.chat_message("assistant"):
            with st.spinner("Agents working..."):
                tracer = AgentTracer()

                try:
                    if mode == "Single Agent" and selected_agent:
                        skill = agents[selected_agent]
                        agent = build_single_agent(skill, TOOL_REGISTRY, tracer=tracer)
                        tracer.start(selected_agent, prompt)
                        result = agent.invoke({"messages": [("user", prompt)]})
                        tracer.end()
                    else:
                        graph = build_multi_agent_graph(
                            agents,
                            TOOL_REGISTRY,
                            tracer=tracer,
                        )
                        result = graph.invoke({
                            "messages": [("user", prompt)],
                            "next_agent": "",
                            "context": {},
                        })

                    # Extract final response
                    final_content = ""
                    for msg in reversed(result["messages"]):
                        if hasattr(msg, "content") and msg.content:
                            content = msg.content if isinstance(msg.content, str) else str(msg.content)
                            if content and not content.startswith("["):
                                final_content = content
                                break

                    if not final_content:
                        final_content = "No response generated. Check the call tree for details."

                    st.markdown(final_content)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": final_content,
                    })

                except Exception as e:
                    error_msg = f"Error: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                    })

                # Save trace
                st.session_state.traces.append((prompt, tracer))

        st.rerun()


# ---- Debug / Call Tree panel ----
with debug_col:
    st.header("Call Tree")

    if st.session_state.traces:
        # Show traces in reverse order (latest first)
        for i, (query, tracer) in enumerate(reversed(st.session_state.traces)):
            trace_idx = len(st.session_state.traces) - 1 - i
            with st.expander(
                f"Run #{trace_idx + 1}: {query[:50]}{'...' if len(query) > 50 else ''}",
                expanded=(i == 0),  # Latest expanded by default
            ):
                # Summary metrics
                render_trace_summary(tracer)

                st.divider()

                # Call tree
                trace_dict = tracer.to_dict()
                if trace_dict:
                    render_trace_node(trace_dict)
                else:
                    st.info("No trace data captured.")

                # Raw trace (collapsible)
                with st.expander("Raw trace JSON", expanded=False):
                    st.json(trace_dict or {})
    else:
        st.info("Send a message to see the call tree here.")
        st.markdown("""
        The call tree shows:
        - **Agent routing** — which agents the supervisor picks
        - **Tool calls** — what tools each agent uses and results
        - **Timing** — how long each step takes
        - **Errors** — any failures with details
        """)
