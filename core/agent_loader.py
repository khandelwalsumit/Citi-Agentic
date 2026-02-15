"""
Agent loader — reads agent definitions from .md skill files.

Each agent is defined by a markdown file in the agents/ directory with
YAML frontmatter for config and markdown body for the system prompt.

Example agents/researcher.md:

    ---
    name: researcher
    description: Gathers information from available sources
    model: gemini-2.5-flash
    temperature: 0.3
    tools:
      - search_documents
      - query_database
    handoffs:
      - analyst
      - writer
    ---

    # System Prompt

    You are a research specialist at Citi Bank.
    Your job is to gather comprehensive information...

To add a new agent: just create a new .md file in the agents/ folder.
To update an agent: edit its .md file. No code changes needed.
"""

from __future__ import annotations

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentSkill:
    """Parsed agent definition from a skill.md file."""

    name: str
    description: str
    system_prompt: str
    model: str = "gemini-2.5-flash"
    temperature: float = 0.0
    max_tokens: int = 8192
    tools: list[str] = field(default_factory=list)
    handoffs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    source_file: str = ""


def _parse_skill_md(filepath: str | Path) -> AgentSkill:
    """Parse a skill.md file into an AgentSkill."""
    filepath = Path(filepath)
    text = filepath.read_text(encoding="utf-8")

    # Split YAML frontmatter from body
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            frontmatter_raw = parts[1].strip()
            body = parts[2].strip()
        else:
            frontmatter_raw = ""
            body = text
    else:
        frontmatter_raw = ""
        body = text

    # Parse frontmatter
    frontmatter: dict = yaml.safe_load(frontmatter_raw) if frontmatter_raw else {}

    return AgentSkill(
        name=frontmatter.get("name", filepath.stem),
        description=frontmatter.get("description", ""),
        system_prompt=body,
        model=frontmatter.get("model", "gemini-2.5-flash"),
        temperature=frontmatter.get("temperature", 0.0),
        max_tokens=frontmatter.get("max_tokens", 8192),
        tools=frontmatter.get("tools", []),
        handoffs=frontmatter.get("handoffs", []),
        metadata=frontmatter.get("metadata", {}),
        source_file=str(filepath),
    )


def load_agent(filepath: str | Path) -> AgentSkill:
    """Load a single agent from a skill.md file."""
    return _parse_skill_md(filepath)


def load_all_agents(agents_dir: str | Path = "agents") -> dict[str, AgentSkill]:
    """Load all agent definitions from a directory.

    Returns a dict keyed by agent name.
    """
    agents_dir = Path(agents_dir)
    if not agents_dir.exists():
        raise FileNotFoundError(f"Agents directory not found: {agents_dir}")

    agents = {}
    for f in sorted(agents_dir.glob("*.md")):
        skill = _parse_skill_md(f)
        agents[skill.name] = skill

    return agents
