```python
"""
planner_executor.py
-------------------
Planner-Executor style LangGraph workflow.

Flow:
  START
    ↓
  filter_node        → calls tools, gets granular themes + their raw data
    ↓
  planner_node       → LLM decides which themes to analyse and in what order
    ↓
  executor_node      → LLM deep-dives ONE granular theme at a time:
    ↓   ↑               - exact digital failure point
    ↓   └─ loop          - root cause in digital journey
    ↓     (until all     - single most actionable fix + owner
    ↓      themes done)
  aggregator_node    → combines all per-theme results → final ranked report
    ↓
  END

Usage:
  result = graph.invoke(
      {"user_query": "top costco signon issues"},
      config={"configurable": {"thread_id": "session-1"}}
  )
  print(result["final_report"])
"""

import json
import re
from typing import TypedDict, Annotated
import operator

import pandas as pd
from langchain_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

import config
from tools_and_prompts import get_filtering_options, filter_contextual_data


# ══════════════════════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════════════════════

class ThemeAnalysis(TypedDict):
    granular_theme:     str
    call_count:         int
    digital_failure:    str   # exact specific digital failure point
    root_cause:         str   # why this specific thing is failing digitally
    actionable_fix:     str   # single most actionable recommendation
    fix_owner:          str   # UI | Feature | Ops | Education
    fix_rationale:      str   # why this fix type
    evidence:           str   # best verbatim example from friction_driver_digital


class PlannerExecutorState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    user_query:         str

    # ── Set by filter_node ────────────────────────────────────────────────────
    filters_applied:    dict          # {product: "...", call_theme: "..."}
    raw_themes:         dict          # raw output from filter_contextual_data tool

    # ── Set by planner_node ───────────────────────────────────────────────────
    plan:               list          # ordered list of granular_theme names to analyse
    plan_rationale:     str           # why this order / what the planner noticed

    # ── Managed by executor loop ──────────────────────────────────────────────
    current_index:      int           # which theme we are currently executing
    theme_results:      Annotated[list, operator.add]  # accumulates ThemeAnalysis dicts

    # ── Set by aggregator_node ────────────────────────────────────────────────
    final_report:       str


# ══════════════════════════════════════════════════════════════════════════════
# LLM
# ══════════════════════════════════════════════════════════════════════════════

llm = ChatVertexAI(model="gemini-1.5-pro", temperature=0)

tools      = [get_filtering_options, filter_contextual_data]
tool_map   = {t.name: t for t in tools}
llm_tools  = llm.bind_tools(tools)


def _parse_json(text: str) -> dict:
    """Strip markdown fences and parse JSON from LLM response."""
    cleaned = re.sub(r"```json\s*|\s*```", "", text).strip()
    return json.loads(cleaned)


def _call(system: str, user: str) -> str:
    """Simple single LLM call, returns text."""
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return resp.content


# ══════════════════════════════════════════════════════════════════════════════
# NODE 1 — FILTER NODE
# Uses tools to translate user query → filters → raw granular theme data
# ══════════════════════════════════════════════════════════════════════════════

FILTER_SYSTEM = """You are a data filter agent. Your only job is to call the available tools
to retrieve the right slice of customer call data for the user's query.

Step 1: Call get_filtering_options to see valid product and call_theme values.
Step 2: Match the user's words to exact values (fuzzy match — "costco" → "Costco Anywhere Visa").
Step 3: Call filter_contextual_data with the matched values.
Step 4: Return ONLY a JSON object with these keys (no other text):
{
  "filters_applied": {"product": "...", "call_theme": "..."},
  "raw_themes": { ...the dict returned by filter_contextual_data... }
}

If the user says "all products" or doesn't mention a product, use empty string for product.
If the user says "all issues" or doesn't mention a theme, use empty string for call_theme."""


def filter_node(state: PlannerExecutorState) -> dict:
    """
    Calls tools to get filtered granular theme data.
    Loops on tool calls until data is retrieved.
    """
    print("\n[Filter] Identifying filters and retrieving data...")
    messages = [
        SystemMessage(content=FILTER_SYSTEM),
        HumanMessage(content=f"User query: {state['user_query']}")
    ]

    # Agentic tool loop — keeps calling tools until LLM returns plain JSON
    while True:
        response = llm_tools.invoke(messages)
        messages.append(response)

        # If no tool calls → LLM has returned the final JSON
        if not getattr(response, "tool_calls", None):
            break

        # Execute each tool call and append results
        for tc in response.tool_calls:
            tool_fn = tool_map[tc["name"]]
            result  = tool_fn.invoke(tc["args"])
            messages.append({
                "role":           "tool",
                "tool_call_id":   tc["id"],
                "content":        json.dumps(result, default=str),
            })

    # Parse the final JSON response
    try:
        parsed = _parse_json(response.content)
        filters = parsed.get("filters_applied", {})
        raw     = parsed.get("raw_themes", {})
        print(f"[Filter] Filters: {filters} | Themes returned: {len(raw)}")
        return {"filters_applied": filters, "raw_themes": raw}
    except Exception as e:
        print(f"[Filter] Parse error: {e} — raw response: {response.content[:200]}")
        return {"filters_applied": {}, "raw_themes": {}}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 2 — PLANNER NODE
# Looks at all granular themes, decides which to analyse and in what order
# ══════════════════════════════════════════════════════════════════════════════

PLANNER_SYSTEM = """You are a planning agent for a customer pain point analysis system.

You will receive a dict of granular pain point themes from customer calls.
Each entry has: granular_theme, call_counts, friction_driver_digital (list), key_solution (list).

Your job:
1. Review ALL themes
2. Select the top themes to deep-analyse (max 8, min 3 — skip themes with call_counts < 10)
3. Order them by priority: call_counts DESC (highest volume = highest priority)
4. Note any patterns you see across themes before the analysis begins

Output ONLY valid JSON — no other text:
{
  "plan": ["granular_theme_name_1", "granular_theme_name_2", ...],
  "plan_rationale": "2-3 sentences: what you noticed, why this order, any cross-theme patterns"
}

The theme names in "plan" must EXACTLY match the granular_theme values in the input."""


def planner_node(state: PlannerExecutorState) -> dict:
    """
    Reviews all granular themes and creates an ordered execution plan.
    """
    raw_themes = state.get("raw_themes", {})
    if not raw_themes:
        return {"plan": [], "plan_rationale": "No data to analyse.", "current_index": 0}

    # Format themes for the planner prompt
    themes_summary = "\n".join([
        f"- {v.get('granular_theme', k)} | calls: {v.get('call_counts', 0)} "
        f"| friction_drivers: {v.get('friction_driver_digital', [])[:3]}"
        for k, v in raw_themes.items()
    ])

    print(f"\n[Planner] Planning analysis for {len(raw_themes)} themes...")

    response = _call(
        system=PLANNER_SYSTEM,
        user=(
            f"User query: {state['user_query']}\n\n"
            f"Available granular themes:\n{themes_summary}"
        )
    )

    try:
        parsed = _parse_json(response)
        plan   = parsed.get("plan", [])
        print(f"[Planner] Plan: {len(plan)} themes to analyse")
        print(f"[Planner] Rationale: {parsed.get('plan_rationale', '')[:100]}...")
        return {
            "plan":           plan,
            "plan_rationale": parsed.get("plan_rationale", ""),
            "current_index":  0,
            "theme_results":  [],
        }
    except Exception as e:
        print(f"[Planner] Parse error: {e}")
        # Fallback: just take all themes sorted by call_counts
        fallback = sorted(
            [v.get("granular_theme", k) for k, v in raw_themes.items()],
            key=lambda t: next(
                (v.get("call_counts", 0) for v in raw_themes.values() if v.get("granular_theme") == t), 0
            ),
            reverse=True
        )[:8]
        return {"plan": fallback, "plan_rationale": "Auto-ranked by call volume.", "current_index": 0, "theme_results": []}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3 — EXECUTOR NODE
# Deep-dives ONE granular theme at a time
# ══════════════════════════════════════════════════════════════════════════════

EXECUTOR_SYSTEM = """You are a Digital Friction Analyst. You analyse ONE specific customer pain point theme
and extract the exact digital failure and the single most actionable fix.

You will receive:
- The granular theme name (the specific cluster label)
- call_counts: how many customers called about this
- friction_driver_digital: list of digital touchpoints/failures that caused the calls
- key_solution: list of recommended solutions

YOUR RULES:
1. digital_failure must be SPECIFIC — not "clarity issue" but "cash advance fee labelled as
   'cash withdrawal' — customer cannot distinguish fee from original transaction"
2. root_cause must explain the exact digital mechanism: label logic, missing link, no notification, etc.
3. actionable_fix must be ONE thing a team can do next sprint — not a vague improvement
4. fix_owner must be exactly one of: UI | Feature | Ops | Education
5. evidence must be the single most specific verbatim item from friction_driver_digital

Output ONLY valid JSON — no other text:
{
  "digital_failure":  "specific 1-sentence description of the exact digital failure",
  "root_cause":       "1 sentence: the underlying digital/system reason this fails",
  "actionable_fix":   "1 sentence: the single most impactful concrete fix",
  "fix_owner":        "UI | Feature | Ops | Education",
  "fix_rationale":    "1 sentence: why this fix type resolves the root cause",
  "evidence":         "best single verbatim item from friction_driver_digital list"
}"""


def executor_node(state: PlannerExecutorState) -> dict:
    """
    Analyses ONE granular theme from the plan (current_index).
    Appends result to theme_results and increments current_index.
    """
    plan          = state["plan"]
    index         = state["current_index"]
    raw_themes    = state["raw_themes"]
    theme_name    = plan[index]

    # Find the matching theme data in raw_themes
    theme_data = next(
        (v for v in raw_themes.values() if v.get("granular_theme") == theme_name),
        None
    )

    if not theme_data:
        print(f"[Executor {index+1}/{len(plan)}] '{theme_name}' — data not found, skipping")
        return {"current_index": index + 1}

    call_count = theme_data.get("call_counts", 0)
    print(f"[Executor {index+1}/{len(plan)}] Analysing: '{theme_name}' ({call_count} calls)...")

    # Deduplicate friction drivers and solutions for cleaner LLM input
    friction_drivers = list(dict.fromkeys(
        str(d) for d in (theme_data.get("friction_driver_digital") or []) if d
    ))
    solutions = list(dict.fromkeys(
        str(s) for s in (theme_data.get("key_solution") or []) if s
    ))

    user_prompt = (
        f"Granular Theme: {theme_name}\n"
        f"Call Count: {call_count}\n\n"
        f"Digital Friction Drivers (what in the digital experience caused these calls):\n"
        + "\n".join(f"  - {d}" for d in friction_drivers[:10])
        + f"\n\nKey Solutions (from agent notes on these calls):\n"
        + "\n".join(f"  - {s}" for s in solutions[:8])
    )

    response = _call(system=EXECUTOR_SYSTEM, user=user_prompt)

    try:
        parsed = _parse_json(response)
        result: ThemeAnalysis = {
            "granular_theme":  theme_name,
            "call_count":      call_count,
            "digital_failure": parsed.get("digital_failure", ""),
            "root_cause":      parsed.get("root_cause", ""),
            "actionable_fix":  parsed.get("actionable_fix", ""),
            "fix_owner":       parsed.get("fix_owner", ""),
            "fix_rationale":   parsed.get("fix_rationale", ""),
            "evidence":        parsed.get("evidence", ""),
        }
        print(f"  → Fix: [{result['fix_owner']}] {result['actionable_fix'][:80]}...")
    except Exception as e:
        print(f"  [Executor] Parse error: {e}")
        result: ThemeAnalysis = {
            "granular_theme":  theme_name,
            "call_count":      call_count,
            "digital_failure": "Parse error — review raw data",
            "root_cause":      str(e),
            "actionable_fix":  "",
            "fix_owner":       "",
            "fix_rationale":   "",
            "evidence":        "",
        }

    return {
        "theme_results": [result],      # Annotated[list, operator.add] accumulates these
        "current_index": index + 1,
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 4 — AGGREGATOR NODE
# Combines all per-theme results into the final ranked report
# ══════════════════════════════════════════════════════════════════════════════

AGGREGATOR_SYSTEM = """You are a report writer for a digital customer experience team.

You receive a list of analysed granular pain point themes, each with:
- granular_theme, call_count, digital_failure, root_cause, actionable_fix, fix_owner, evidence

Write a clear, structured insight report. Follow this EXACT format:

---
## Digital Friction Report
**Query:** {query}
**Filters:** {filters}
**Themes analysed:** {n} | **Total calls in view:** {total}

---
### #{rank} · {granular_theme} · {call_count} calls
**Digital Failure:** {digital_failure}
**Root Cause:** {root_cause}
**Actionable Fix [{fix_owner}]:** {actionable_fix}
*Why this fix:* {fix_rationale}
**Evidence:** "{evidence}"

---
[repeat for each theme, ranked by call_count DESC]

## Cross-Theme Patterns
[3-5 bullet points on patterns you see across ALL themes:
 - Is there a common root cause type (labelling? notifications? navigation?)
 - Which fix_owner appears most? What does that tell the team?
 - Any quick wins (UI fixes that would resolve multiple themes)?
 - Any systemic issues that need a Feature investment?]

## Top 3 Immediate Actions
1. [highest impact, lowest effort action]
2. [second]
3. [third]
---

Rules:
- digital_failure must stay specific — do not generalise it
- Keep each section tight — product teams read this in 2 minutes
- Cross-Theme Patterns is the most valuable section — be sharp"""


def aggregator_node(state: PlannerExecutorState) -> dict:
    """
    Combines all executor results into the final report.
    Sorts by call_count descending before passing to LLM.
    """
    results = state.get("theme_results", [])
    if not results:
        return {"final_report": "No themes were analysed. Check filters and data."}

    # Sort by call_count descending for the report
    results_sorted = sorted(results, key=lambda x: x.get("call_count", 0), reverse=True)
    total_calls    = sum(r.get("call_count", 0) for r in results_sorted)
    filters        = state.get("filters_applied", {})

    print(f"\n[Aggregator] Writing report for {len(results_sorted)} themes ({total_calls:,} total calls)...")

    # Add rank numbers
    numbered = [
        {**r, "rank": i + 1}
        for i, r in enumerate(results_sorted)
    ]

    user_prompt = (
        f"Query: {state['user_query']}\n"
        f"Filters: {json.dumps(filters)}\n"
        f"Total calls in view: {total_calls:,}\n\n"
        f"Analysed themes (sorted by call volume):\n"
        f"{json.dumps(numbered, indent=2)}"
    )

    report = _call(system=AGGREGATOR_SYSTEM, user=user_prompt)

    print("[Aggregator] Report complete.")
    return {"final_report": report}


# ══════════════════════════════════════════════════════════════════════════════
# ROUTING
# ══════════════════════════════════════════════════════════════════════════════

def should_continue_executing(state: PlannerExecutorState) -> str:
    """
    After each executor run:
      - More themes left in the plan → loop back to executor
      - All themes done             → go to aggregator
    """
    if state["current_index"] < len(state["plan"]):
        return "executor_node"
    return "aggregator_node"


# ══════════════════════════════════════════════════════════════════════════════
# BUILD GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def build_graph():
    graph = StateGraph(PlannerExecutorState)

    graph.add_node("filter_node",     filter_node)
    graph.add_node("planner_node",    planner_node)
    graph.add_node("executor_node",   executor_node)
    graph.add_node("aggregator_node", aggregator_node)

    graph.add_edge(START,              "filter_node")
    graph.add_edge("filter_node",      "planner_node")
    graph.add_edge("planner_node",     "executor_node")

    # Executor loops until all themes in the plan are done
    graph.add_conditional_edges(
        "executor_node",
        should_continue_executing,
        {
            "executor_node":   "executor_node",
            "aggregator_node": "aggregator_node",
        }
    )

    graph.add_edge("aggregator_node", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


graph = build_graph()


# ══════════════════════════════════════════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════════════════════════════════════════

def run(query: str, thread_id: str = "default") -> str:
    """
    Run a natural language pain point query through the planner-executor graph.

    Args:
        query:     e.g. "top costco signon issues"
        thread_id: reuse to continue a conversation, new id for fresh session

    Returns:
        The final markdown report as a string.
    """
    initial_state: PlannerExecutorState = {
        "user_query":      query,
        "filters_applied": {},
        "raw_themes":      {},
        "plan":            [],
        "plan_rationale":  "",
        "current_index":   0,
        "theme_results":   [],
        "final_report":    "",
    }

    result = graph.invoke(
        initial_state,
        config={"configurable": {"thread_id": thread_id}}
    )

    return result["final_report"]


# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(run("top issues in costco for sign on", thread_id="demo-1"))
