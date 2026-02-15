"""
Citi Agentic — Multi-Agent System powered by skill.md definitions.

Usage:
    # Single agent mode (for testing)
    python main.py --agent researcher "Find Q4 2025 earnings data"

    # Multi-agent mode (full pipeline)
    python main.py "Analyze the impact of recent Fed rate changes on our ICG portfolio"

    # Interactive mode
    python main.py --interactive
"""

import argparse
import os
import sys
import json

import config
from core.auth import get_api_key, init_vertex
from core.chat_model import CitiVertexChat
from core.agent_loader import load_all_agents, load_agent
from core.orchestrator import build_multi_agent_graph, build_single_agent
from tools.common import TOOL_REGISTRY


def setup_auth():
    """Initialize R2D2 Vertex AI authentication."""
    os.environ["REQUESTS_CA_BUNDLE"] = config.CA_BUNDLE

    if not config.COIN_CLIENT_ID:
        print("ERROR: Set COIN_CLIENT_ID, COIN_CLIENT_SECRET, COIN_CLIENT_SCOPES in .env")
        sys.exit(1)

    print("Authenticating with R2D2...")
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
    print("Authenticated successfully.\n")


def run_single_agent(agent_name: str, query: str):
    """Run a single agent by name."""
    agents = load_all_agents(config.AGENTS_DIR)
    if agent_name not in agents:
        print(f"ERROR: Agent '{agent_name}' not found. Available: {list(agents.keys())}")
        sys.exit(1)

    skill = agents[agent_name]
    print(f"Running agent: {skill.name} ({skill.description})")
    print(f"Model: {skill.model} | Tools: {skill.tools}")
    print("-" * 60)

    agent = build_single_agent(skill, TOOL_REGISTRY)
    result = agent.invoke({"messages": [("user", query)]})

    # Print the final response
    for msg in result["messages"]:
        if hasattr(msg, "content") and msg.content:
            role = msg.__class__.__name__.replace("Message", "")
            print(f"\n[{role}]: {msg.content}")


def run_multi_agent(query: str):
    """Run the full multi-agent pipeline."""
    agents = load_all_agents(config.AGENTS_DIR)
    print(f"Loaded {len(agents)} agents: {list(agents.keys())}")
    print("-" * 60)

    graph = build_multi_agent_graph(
        agents,
        TOOL_REGISTRY,
        supervisor_model=config.DEFAULT_MODEL,
    )

    result = graph.invoke({
        "messages": [("user", query)],
        "next_agent": "",
        "context": {},
    })

    # Print final output
    print("\n" + "=" * 60)
    print("FINAL OUTPUT")
    print("=" * 60)
    for msg in result["messages"]:
        if hasattr(msg, "content") and msg.content:
            role = msg.__class__.__name__.replace("Message", "")
            print(f"\n[{role}]: {msg.content}")


def run_interactive():
    """Run in interactive chat mode with the multi-agent system."""
    agents = load_all_agents(config.AGENTS_DIR)
    print(f"Loaded {len(agents)} agents: {list(agents.keys())}")

    graph = build_multi_agent_graph(
        agents,
        TOOL_REGISTRY,
        supervisor_model=config.DEFAULT_MODEL,
    )

    print("\nCiti Agentic System — Interactive Mode")
    print("Type 'quit' to exit, 'agents' to list agents\n")

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not query:
            continue
        if query.lower() == "quit":
            break
        if query.lower() == "agents":
            for name, skill in agents.items():
                print(f"  {name}: {skill.description} (tools: {skill.tools})")
            continue

        result = graph.invoke({
            "messages": [("user", query)],
            "next_agent": "",
            "context": {},
        })

        # Show the last AI message
        for msg in reversed(result["messages"]):
            if hasattr(msg, "content") and msg.content and "AI" in msg.__class__.__name__:
                print(f"\nAssistant: {msg.content}\n")
                break


def main():
    parser = argparse.ArgumentParser(description="Citi Agentic Multi-Agent System")
    parser.add_argument("query", nargs="?", help="The task/question for the agents")
    parser.add_argument("--agent", "-a", help="Run a single agent by name")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--list-agents", "-l", action="store_true", help="List available agents")
    args = parser.parse_args()

    if args.list_agents:
        agents = load_all_agents(config.AGENTS_DIR)
        for name, skill in agents.items():
            print(f"  {name}:")
            print(f"    description: {skill.description}")
            print(f"    model: {skill.model}")
            print(f"    tools: {skill.tools}")
            print(f"    handoffs: {skill.handoffs}")
            print(f"    file: {skill.source_file}")
            print()
        return

    # Auth required for all modes that call the model
    setup_auth()

    if args.interactive:
        run_interactive()
    elif args.agent and args.query:
        run_single_agent(args.agent, args.query)
    elif args.query:
        run_multi_agent(args.query)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
