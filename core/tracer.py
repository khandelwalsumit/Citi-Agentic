"""
Agent tracer — captures the full call tree for debugging.

Records every agent invocation, supervisor routing decision, tool call,
and message flow. Designed to feed into the Streamlit debug panel.

Usage:
    tracer = AgentTracer()
    result = graph.invoke(input, config={"callbacks": [tracer]})
    # Or use the traced orchestrator wrapper
    tracer.print_tree()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ToolCallTrace:
    """Record of a single tool invocation."""
    name: str
    args: dict
    result: str = ""
    duration_ms: float = 0.0


@dataclass
class AgentTrace:
    """Record of a single agent invocation within the call tree."""
    agent_name: str
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    input_summary: str = ""
    output_summary: str = ""
    tool_calls: list[ToolCallTrace] = field(default_factory=list)
    routing_decision: str = ""  # For supervisor: which agent it picked
    children: list[AgentTrace] = field(default_factory=list)
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000

    @property
    def status(self) -> str:
        if self.error:
            return "error"
        if self.end_time is None:
            return "running"
        return "done"

    @property
    def timestamp(self) -> str:
        return datetime.fromtimestamp(self.start_time).strftime("%H:%M:%S.%f")[:-3]

    def to_dict(self) -> dict:
        """Serialize for Streamlit display."""
        return {
            "agent": self.agent_name,
            "status": self.status,
            "timestamp": self.timestamp,
            "duration_ms": round(self.duration_ms, 1),
            "input": self.input_summary[:200],
            "output": self.output_summary[:200],
            "routing": self.routing_decision,
            "tool_calls": [
                {"name": tc.name, "args": tc.args, "result": tc.result[:100], "ms": tc.duration_ms}
                for tc in self.tool_calls
            ],
            "error": self.error,
            "children": [c.to_dict() for c in self.children],
        }


class AgentTracer:
    """Captures the full execution trace of a multi-agent run.

    Thread-safe for single-graph execution. Create a new instance per run.
    """

    def __init__(self):
        self.root: AgentTrace | None = None
        self._stack: list[AgentTrace] = []
        self.all_traces: list[AgentTrace] = []  # flat list of all traces in order

    def start(self, agent_name: str, input_summary: str = "") -> AgentTrace:
        """Record the start of an agent invocation."""
        trace = AgentTrace(
            agent_name=agent_name,
            input_summary=input_summary,
        )
        if self._stack:
            self._stack[-1].children.append(trace)
        else:
            self.root = trace

        self._stack.append(trace)
        self.all_traces.append(trace)
        return trace

    def end(self, output_summary: str = "", error: str | None = None):
        """Record the end of the current agent invocation."""
        if not self._stack:
            return
        trace = self._stack.pop()
        trace.end_time = time.time()
        trace.output_summary = output_summary
        trace.error = error

    def log_tool_call(self, name: str, args: dict, result: str = "", duration_ms: float = 0.0):
        """Log a tool call within the current agent."""
        if not self._stack:
            return
        self._stack[-1].tool_calls.append(
            ToolCallTrace(name=name, args=args, result=result, duration_ms=duration_ms)
        )

    def log_routing(self, decision: str):
        """Log a supervisor routing decision."""
        if not self._stack:
            return
        self._stack[-1].routing_decision = decision

    @property
    def total_duration_ms(self) -> float:
        return self.root.duration_ms if self.root else 0.0

    @property
    def agent_count(self) -> int:
        return len(self.all_traces)

    @property
    def tool_call_count(self) -> int:
        return sum(len(t.tool_calls) for t in self.all_traces)

    def to_dict(self) -> dict | None:
        """Full trace tree as a dict for Streamlit."""
        return self.root.to_dict() if self.root else None

    def print_tree(self, node: AgentTrace | None = None, indent: int = 0):
        """Print the call tree to stdout (for CLI debugging)."""
        if node is None:
            node = self.root
        if node is None:
            print("(empty trace)")
            return

        prefix = "  " * indent
        status_icon = {"done": "OK", "running": "...", "error": "ERR"}.get(node.status, "?")
        duration = f"{node.duration_ms:.0f}ms"

        print(f"{prefix}[{status_icon}] {node.agent_name} ({duration})")

        if node.routing_decision:
            print(f"{prefix}  -> routed to: {node.routing_decision}")

        for tc in node.tool_calls:
            print(f"{prefix}  * {tc.name}({tc.args}) = {tc.result[:80]}")

        if node.error:
            print(f"{prefix}  ERROR: {node.error}")

        for child in node.children:
            self.print_tree(child, indent + 1)
