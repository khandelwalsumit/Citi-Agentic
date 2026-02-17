
START
  ↓
filter_node      → tool loop: get valid filters → filter data → raw granular themes
  ↓
planner_node     → reviews ALL themes, picks top 3-8, orders by volume, notes patterns
  ↓
executor_node  ←─────────────────────────────────────────┐
  ↓                                                       │
  → analyses ONE theme (digital_failure + root_cause      │
    + actionable_fix + fix_owner + evidence)              │
  ↓                                                       │
should_continue_executing ──── more themes? ─────────────┘
  ↓ all done
aggregator_node  → combines all results → cross-theme patterns → top 3 actions
  ↓
END

```python

"""
adaptive_planner_executor.py
-----------------------------
Intelligent planner-executor with:
  1. Adaptive hierarchy navigation (drill down on high-volume themes)
  2. Theme-specific specialized agents (payment vs signon vs transaction analysis)
  3. Multi-dimensional prioritization (urgency × ease × volume)

Flow:
  filter_node            → get raw data
    ↓
  hierarchy_navigator    → adaptive rollup/rolldown
    ├─ start at broad_theme level
    ├─ if volume > 200 → drill to intermediate_theme
    ├─ if still > 200  → drill to granular_theme
    └─ output: mixed-level themes ready for analysis
    ↓
  theme_router          → assign each theme to specialized agent
    ↓
  [parallel executors]  → payment_agent | signon_agent | transaction_agent | ...
    ↓
  prioritizer           → rank by urgency × ease × volume
    ↓
  report_generator      → final report
    ↓
  END
"""

import json
import re
from typing import TypedDict, Literal, Annotated
import operator

import pandas as pd
from langchain_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, Send
from langgraph.checkpoint.memory import MemorySaver

import config
from tools_and_prompts import get_filtering_options, filter_contextual_data


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

VOLUME_THRESHOLD = 200  # if theme volume > this, drill down to next level

SPECIALIST_MAP = {
    "payment":       "payment_specialist",
    "transaction":   "transaction_specialist",
    "signon":        "auth_specialist",
    "reward":        "rewards_specialist",
    "dispute/fraud": "dispute_specialist",
    "card":          "card_specialist",
    "other":         "general_specialist",
}


# ══════════════════════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════════════════════

class ThemeForAnalysis(TypedDict):
    theme_name:        str           # the theme label (at any hierarchy level)
    theme_level:       str           # "broad" | "intermediate" | "granular"
    call_count:        int
    dominant_call_reason: str        # used for routing to specialist
    friction_drivers:  list          # raw friction_driver_digital texts
    key_solutions:     list          # raw key_solution texts
    specialist_type:   str           # which specialist will handle this


class ThemeAnalysisResult(TypedDict):
    theme_name:        str
    theme_level:       str
    call_count:        int
    digital_failure:   str           # specific failure
    root_cause:        str           # why it fails
    actionable_fix:    str           # single concrete fix
    fix_owner:         str           # UI | Feature | Ops | Education
    fix_rationale:     str           # why this fix
    urgency_score:     int           # 1-5: how critical
    ease_score:        int           # 1-5: how easy to fix (5 = easiest)
    priority_score:    float         # urgency × ease × (volume/1000)
    evidence:          str           # best verbatim friction_driver


class AdaptiveState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    user_query:         str

    # ── Filter node ───────────────────────────────────────────────────────────
    filters_applied:    dict
    raw_data:           pd.DataFrame  # full dataset after filters (for adaptive navigation)

    # ── Hierarchy navigator ───────────────────────────────────────────────────
    themes_for_analysis: list[ThemeForAnalysis]  # themes ready for specialist analysis
    navigation_log:     str                       # audit trail of drill-down decisions

    # ── Parallel executors ────────────────────────────────────────────────────
    analysis_results:   Annotated[list[ThemeAnalysisResult], operator.add]

    # ── Prioritizer ───────────────────────────────────────────────────────────
    prioritized_results: list[ThemeAnalysisResult]

    # ── Report ────────────────────────────────────────────────────────────────
    final_report:       str


# ══════════════════════════════════════════════════════════════════════════════
# LLM
# ══════════════════════════════════════════════════════════════════════════════

llm = ChatVertexAI(model="gemini-1.5-pro", temperature=0)

tools     = [get_filtering_options, filter_contextual_data]
tool_map  = {t.name: t for t in tools}
llm_tools = llm.bind_tools(tools)


def _call(system: str, user: str) -> str:
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return resp.content


def _parse_json(text: str) -> dict:
    cleaned = re.sub(r"```json\s*|\s*```", "", text).strip()
    return json.loads(cleaned)


# ══════════════════════════════════════════════════════════════════════════════
# NODE 1 — FILTER NODE
# ══════════════════════════════════════════════════════════════════════════════

FILTER_SYSTEM = """You are a data filter agent. Call tools to retrieve customer call data.

Step 1: Call get_filtering_options to see valid values.
Step 2: Match user's words to exact values (fuzzy match).
Step 3: Call filter_contextual_data with matched values.
Step 4: Return the raw result from filter_contextual_data as-is."""


def filter_node(state: AdaptiveState) -> dict:
    """Calls tools to get filtered data. Returns raw pandas-compatible dict."""
    print(f"\n[Filter] Processing query: {state['user_query']}")

    messages = [
        SystemMessage(content=FILTER_SYSTEM),
        HumanMessage(content=f"User query: {state['user_query']}")
    ]

    # Tool loop
    while True:
        response = llm_tools.invoke(messages)
        messages.append(response)

        if not getattr(response, "tool_calls", None):
            break

        for tc in response.tool_calls:
            result = tool_map[tc["name"]].invoke(tc["args"])
            messages.append({
                "role": "tool", "tool_call_id": tc["id"],
                "content": json.dumps(result, default=str)
            })

    # Load the full filtered dataset from config.DATA_FILE for adaptive navigation
    # (tool only returns top 10, we need the full dataset for hierarchy analysis)
    data = pd.read_csv(config.DATA_FILE)

    # Apply same filters that the tool used
    # (parse the last message to extract filters — or reuse the tool result logic)
    # For simplicity, assume user query maps to product/call_theme
    # You can extract from messages or re-call filter logic here

    print(f"[Filter] Dataset loaded: {len(data):,} total rows")
    return {"raw_data": data, "filters_applied": {}}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 2 — HIERARCHY NAVIGATOR (THE ADAPTIVE DRILL-DOWN BRAIN)
# ══════════════════════════════════════════════════════════════════════════════

def hierarchy_navigator(state: AdaptiveState) -> dict:
    """
    Adaptive hierarchy navigation:
    1. Start at broad_theme level, aggregate by volume
    2. For any theme with volume > VOLUME_THRESHOLD, drill to intermediate_theme
    3. For any intermediate with volume > VOLUME_THRESHOLD, drill to granular_theme
    4. Return: list of themes at mixed levels, all ready for specialist analysis
    """
    df = state["raw_data"]
    if df is None or len(df) == 0:
        return {"themes_for_analysis": [], "navigation_log": "No data to analyse."}

    print(f"\n[Navigator] Starting adaptive drill-down (threshold: {VOLUME_THRESHOLD})...")
    log = []
    final_themes = []

    # ── LEVEL 1: Broad Theme ──────────────────────────────────────────────────
    broad = df.groupby("broad_theme").agg(
        volume=("call_id", "count"),
        dominant_call_reason=("call_reason", lambda x: x.mode()[0] if len(x) > 0 else "other"),
    ).reset_index()

    log.append(f"Level 1 (Broad): {len(broad)} themes")

    for _, row in broad.iterrows():
        theme = row["broad_theme"]
        vol   = row["volume"]

        if vol <= VOLUME_THRESHOLD:
            # Volume low enough → keep at broad level
            theme_df = df[df["broad_theme"] == theme]
            final_themes.append(_extract_theme_data(theme_df, theme, "broad", row["dominant_call_reason"]))
            log.append(f"  ✓ '{theme}' ({vol}) → keep at BROAD level")
        else:
            # Volume too high → drill to intermediate
            log.append(f"  ↓ '{theme}' ({vol}) → drill to INTERMEDIATE")
            _drill_intermediate(df, theme, final_themes, log)

    navigation_log = "\n".join(log)
    print(f"[Navigator] Final: {len(final_themes)} themes selected for analysis")
    print(f"[Navigator] Breakdown: {sum(1 for t in final_themes if t['theme_level']=='broad')} broad, "
          f"{sum(1 for t in final_themes if t['theme_level']=='intermediate')} intermediate, "
          f"{sum(1 for t in final_themes if t['theme_level']=='granular')} granular")

    return {"themes_for_analysis": final_themes, "navigation_log": navigation_log}


def _drill_intermediate(df: pd.DataFrame, broad_name: str, final_themes: list, log: list):
    """Drill from broad → intermediate for a specific broad theme."""
    subset = df[df["broad_theme"] == broad_name]
    inter = subset.groupby("intermediate_theme").agg(
        volume=("call_id", "count"),
        dominant_call_reason=("call_reason", lambda x: x.mode()[0] if len(x) > 0 else "other"),
    ).reset_index()

    for _, row in inter.iterrows():
        theme = row["intermediate_theme"]
        vol   = row["volume"]

        if vol <= VOLUME_THRESHOLD:
            theme_df = subset[subset["intermediate_theme"] == theme]
            final_themes.append(_extract_theme_data(theme_df, theme, "intermediate", row["dominant_call_reason"]))
            log.append(f"    ✓ '{theme}' ({vol}) → keep at INTERMEDIATE")
        else:
            log.append(f"    ↓ '{theme}' ({vol}) → drill to GRANULAR")
            _drill_granular(subset, theme, final_themes, log)


def _drill_granular(df: pd.DataFrame, inter_name: str, final_themes: list, log: list):
    """Drill from intermediate → granular for a specific intermediate theme."""
    subset = df[df["intermediate_theme"] == inter_name]
    gran = subset.groupby("granular_theme").agg(
        volume=("call_id", "count"),
        dominant_call_reason=("call_reason", lambda x: x.mode()[0] if len(x) > 0 else "other"),
    ).reset_index()

    for _, row in gran.iterrows():
        theme = row["granular_theme"]
        vol   = row["volume"]
        theme_df = subset[subset["granular_theme"] == theme]
        final_themes.append(_extract_theme_data(theme_df, theme, "granular", row["dominant_call_reason"]))
        log.append(f"      ✓ '{theme}' ({vol}) → GRANULAR level")


def _extract_theme_data(theme_df: pd.DataFrame, theme_name: str, level: str, call_reason: str) -> ThemeForAnalysis:
    """Extract friction drivers and solutions from a theme's calls."""
    return {
        "theme_name":          theme_name,
        "theme_level":         level,
        "call_count":          len(theme_df),
        "dominant_call_reason": call_reason,
        "friction_drivers":    list(theme_df["friction_driver_digital"].dropna().unique())[:15],
        "key_solutions":       list(theme_df["key_solution"].dropna().unique())[:10],
        "specialist_type":     SPECIALIST_MAP.get(call_reason, "general_specialist"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3 — THEME ROUTER (decides which specialist each theme goes to)
# ══════════════════════════════════════════════════════════════════════════════

def theme_router(state: AdaptiveState):
    """
    Fan-out: sends each theme to its specialized executor in parallel.
    Uses Send() to create parallel branches.
    """
    themes = state["themes_for_analysis"]
    print(f"\n[Router] Routing {len(themes)} themes to specialized executors...")

    # Group by specialist type for logging
    by_specialist = {}
    for theme in themes:
        spec = theme["specialist_type"]
        by_specialist[spec] = by_specialist.get(spec, 0) + 1

    for spec, count in by_specialist.items():
        print(f"  → {spec}: {count} themes")

    # Send each theme to the appropriate executor node
    return [
        Send("execute_theme", {"theme": theme})
        for theme in themes
    ]


# ══════════════════════════════════════════════════════════════════════════════
# NODE 4 — SPECIALIZED EXECUTORS (one per call_reason type)
# ══════════════════════════════════════════════════════════════════════════════

PAYMENT_SPECIALIST_SYSTEM = """You are a Payment Flow Specialist. Analyse payment-related friction.

Focus on:
- Payment confirmation gaps (user unsure if payment processed)
- Payment method failures (bank account not saved, payment declined)
- Payment date mismatches (payment posted but not reflected)
- AutoPay setup confusion

Output urgency_score based on:
- 5: Customer cannot complete payment (revenue-blocking)
- 4: Customer can pay but needs to call for confirmation (high call volume driver)
- 3: Payment works but UX is unclear (causes doubt)
- 2: Minor edge case
- 1: Rare issue

Output ease_score based on:
- 5: Simple UI label change or copy update
- 4: UI component addition (button, status indicator)
- 3: Feature flag / config change
- 2: New feature development (notification system, etc.)
- 1: Complex backend system integration"""

SIGNON_SPECIALIST_SYSTEM = """You are an Authentication Specialist. Analyse sign-on / login friction.

Focus on:
- OTP delivery failures (SMS not received, email delayed)
- Password reset flow failures (link expired, link not working)
- Biometric auth failures (Face ID / Touch ID not working)
- Account lockout issues (too many failed attempts)

Output urgency_score based on:
- 5: Customer locked out, cannot access account at all (critical blocker)
- 4: Customer can authenticate but only after multiple attempts / call
- 3: Alternate auth method works but preferred method fails
- 2: Inconvenience but workaround exists
- 1: Rare edge case

Output ease_score based on:
- 5: UI messaging / help text improvement
- 4: Add retry button, resend OTP, status indicator
- 3: Change timeout/expiry policy (config change)
- 2: Implement fallback auth method
- 1: Replace third-party SMS provider"""

TRANSACTION_SPECIALIST_SYSTEM = """You are a Transaction Clarity Specialist. Analyse transaction display issues.

Focus on:
- Transaction labeling (fee shown as withdrawal, merchant name unclear)
- Transaction categorization (cash advance vs purchase)
- Transaction linking (fee not linked to parent transaction)
- Pending vs posted confusion

Output urgency_score based on:
- 5: Customer disputes transaction thinking it's fraud (dispute call volume)
- 4: Customer calls to understand a valid transaction (high volume driver)
- 3: Customer confused but can figure it out
- 2: Cosmetic issue
- 1: Rare scenario

Output ease_score based on:
- 5: Relabel transaction display strings
- 4: Link transactions (fee → parent) in UI
- 3: Add transaction detail drill-down feature
- 2: Merchant name normalization pipeline
- 1: Change transaction categorization logic at processor level"""

GENERAL_SPECIALIST_SYSTEM = """You are a General Digital Experience Specialist.

Analyse any customer friction point and identify:
- The specific digital failure
- The root cause in the digital journey
- A single concrete fix

Rate urgency (1-5) based on customer impact severity.
Rate ease (1-5) based on implementation complexity (5 = easiest).

Default scoring when unsure:
- urgency_score: 3 (moderate impact)
- ease_score: 3 (moderate complexity)"""

SPECIALIST_PROMPTS = {
    "payment_specialist":     PAYMENT_SPECIALIST_SYSTEM,
    "auth_specialist":        SIGNON_SPECIALIST_SYSTEM,
    "transaction_specialist": TRANSACTION_SPECIALIST_SYSTEM,
    "rewards_specialist":     GENERAL_SPECIALIST_SYSTEM,
    "dispute_specialist":     GENERAL_SPECIALIST_SYSTEM,
    "card_specialist":        GENERAL_SPECIALIST_SYSTEM,
    "general_specialist":     GENERAL_SPECIALIST_SYSTEM,
}

EXECUTOR_OUTPUT_FORMAT = """Output ONLY valid JSON — no other text:
{
  "digital_failure":  "specific 1-sentence failure",
  "root_cause":       "1 sentence why it fails",
  "actionable_fix":   "1 sentence concrete fix",
  "fix_owner":        "UI | Feature | Ops | Education",
  "fix_rationale":    "1 sentence why this fix resolves root cause",
  "urgency_score":    1-5 integer,
  "ease_score":       1-5 integer,
  "evidence":         "best verbatim item from friction_drivers"
}"""


def execute_theme(state: dict) -> dict:
    """
    Executes specialist analysis for ONE theme.
    Receives: {"theme": ThemeForAnalysis}
    Returns: {"analysis_results": [ThemeAnalysisResult]}
    """
    theme: ThemeForAnalysis = state["theme"]
    spec_type = theme["specialist_type"]
    system    = SPECIALIST_PROMPTS[spec_type] + "\n\n" + EXECUTOR_OUTPUT_FORMAT

    print(f"[{spec_type}] Analysing: '{theme['theme_name']}' ({theme['call_count']} calls, {theme['theme_level']} level)")

    user_prompt = (
        f"Theme: {theme['theme_name']}\n"
        f"Level: {theme['theme_level']}\n"
        f"Call Count: {theme['call_count']}\n"
        f"Call Reason: {theme['dominant_call_reason']}\n\n"
        f"Friction Drivers:\n" + "\n".join(f"  - {d}" for d in theme["friction_drivers"][:12]) + "\n\n"
        f"Key Solutions:\n" + "\n".join(f"  - {s}" for s in theme["key_solutions"][:8])
    )

    response = _call(system=system, user=user_prompt)

    try:
        parsed = _parse_json(response)
        result: ThemeAnalysisResult = {
            "theme_name":      theme["theme_name"],
            "theme_level":     theme["theme_level"],
            "call_count":      theme["call_count"],
            "digital_failure": parsed.get("digital_failure", ""),
            "root_cause":      parsed.get("root_cause", ""),
            "actionable_fix":  parsed.get("actionable_fix", ""),
            "fix_owner":       parsed.get("fix_owner", ""),
            "fix_rationale":   parsed.get("fix_rationale", ""),
            "urgency_score":   int(parsed.get("urgency_score", 3)),
            "ease_score":      int(parsed.get("ease_score", 3)),
            "priority_score":  0.0,  # computed later by prioritizer
            "evidence":        parsed.get("evidence", ""),
        }
        print(f"  → [{result['fix_owner']}] urgency={result['urgency_score']}, ease={result['ease_score']}")
    except Exception as e:
        print(f"  [ERROR] Parse failed: {e}")
        result: ThemeAnalysisResult = {
            "theme_name":      theme["theme_name"],
            "theme_level":     theme["theme_level"],
            "call_count":      theme["call_count"],
            "digital_failure": f"Parse error: {e}",
            "root_cause":      "",
            "actionable_fix":  "",
            "fix_owner":       "",
            "fix_rationale":   "",
            "urgency_score":   3,
            "ease_score":      3,
            "priority_score":  0.0,
            "evidence":        "",
        }

    return {"analysis_results": [result]}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 5 — PRIORITIZER
# ══════════════════════════════════════════════════════════════════════════════

def prioritizer(state: AdaptiveState) -> dict:
    """
    Ranks all results by: urgency × ease × (volume / 1000)
    Higher priority_score = more urgent + easier + higher volume.
    """
    results = state.get("analysis_results", [])
    if not results:
        return {"prioritized_results": []}

    print(f"\n[Prioritizer] Computing priority scores for {len(results)} themes...")

    for r in results:
        r["priority_score"] = (
            r["urgency_score"] * r["ease_score"] * (r["call_count"] / 1000.0)
        )

    sorted_results = sorted(results, key=lambda x: x["priority_score"], reverse=True)

    print("[Prioritizer] Top 3 by priority:")
    for i, r in enumerate(sorted_results[:3], 1):
        print(f"  {i}. {r['theme_name'][:50]} | priority={r['priority_score']:.1f} "
              f"(U:{r['urgency_score']} × E:{r['ease_score']} × V:{r['call_count']})")

    return {"prioritized_results": sorted_results}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 6 — REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

REPORT_SYSTEM = """You are a report writer. Generate a clear executive report.

Format:
---
## Digital Friction Analysis
**Query:** {query}
**Themes analysed:** {n} (mixed hierarchy: {broad} broad, {intermediate} intermediate, {granular} granular)
**Total call volume:** {total}

---
### #{rank} · {theme_name} ({level} level) · {call_count} calls
**Priority Score:** {priority_score:.1f} (Urgency: {urgency}/5 | Ease: {ease}/5)
**Digital Failure:** {digital_failure}
**Root Cause:** {root_cause}
**Actionable Fix [{fix_owner}]:** {actionable_fix}
*Why:* {fix_rationale}
**Evidence:** "{evidence}"

---
[repeat for all themes]

## Priority Matrix Insights
- **Quick Wins** (high urgency + high ease): [list theme names]
- **Strategic Investments** (high urgency + low ease): [list theme names]
- **Low-Hanging Fruit** (low urgency + high ease): [list theme names]

## Top 3 Immediate Actions
1. [highest priority_score action with owner]
2. [second]
3. [third]
---"""


def report_generator(state: AdaptiveState) -> dict:
    """Generates final markdown report from prioritized results."""
    results = state.get("prioritized_results", [])
    if not results:
        return {"final_report": "No themes analysed."}

    total_calls = sum(r["call_count"] for r in results)
    broad_count = sum(1 for r in results if r["theme_level"] == "broad")
    inter_count = sum(1 for r in results if r["theme_level"] == "intermediate")
    gran_count  = sum(1 for r in results if r["theme_level"] == "granular")

    print(f"\n[Report] Generating final report for {len(results)} themes...")

    numbered = [{"rank": i+1, **r} for i, r in enumerate(results)]

    user_prompt = (
        f"Query: {state['user_query']}\n"
        f"Themes: {len(results)} ({broad_count} broad, {inter_count} intermediate, {gran_count} granular)\n"
        f"Total calls: {total_calls:,}\n\n"
        f"Prioritized themes (sorted by priority_score DESC):\n"
        f"{json.dumps(numbered, indent=2)}"
    )

    report = _call(system=REPORT_SYSTEM, user=user_prompt)
    print("[Report] Complete.")
    return {"final_report": report}


# ══════════════════════════════════════════════════════════════════════════════
# BUILD GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def build_graph():
    graph = StateGraph(AdaptiveState)

    graph.add_node("filter_node",         filter_node)
    graph.add_node("hierarchy_navigator", hierarchy_navigator)
    graph.add_node("execute_theme",       execute_theme)  # parallel executor
    graph.add_node("prioritizer",         prioritizer)
    graph.add_node("report_generator",    report_generator)

    graph.add_edge(START,                  "filter_node")
    graph.add_edge("filter_node",          "hierarchy_navigator")

    # Fan-out to parallel executors via conditional edges + Send()
    graph.add_conditional_edges("hierarchy_navigator", theme_router, ["execute_theme"])

    # All executors converge to prioritizer
    graph.add_edge("execute_theme",       "prioritizer")
    graph.add_edge("prioritizer",         "report_generator")
    graph.add_edge("report_generator",    END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


graph = build_graph()


# ══════════════════════════════════════════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════════════════════════════════════════

def run(query: str, thread_id: str = "default") -> str:
    initial_state: AdaptiveState = {
        "user_query":          query,
        "filters_applied":     {},
        "raw_data":            None,
        "themes_for_analysis": [],
        "navigation_log":      "",
        "analysis_results":    [],
        "prioritized_results": [],
        "final_report":        "",
    }

    result = graph.invoke(initial_state, config={"configurable": {"thread_id": thread_id}})
    return result["final_report"]


if __name__ == "__main__":
    print(run("top issues in costco for sign on"))
