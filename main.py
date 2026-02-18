"""
multi_agent.py
--------------
Adaptive planner-executor with:
  1. Planner node (answer/clarify/execute routing)
  2. Adaptive hierarchy navigation (drill down on high-volume themes)
  3. Rate-limited batch processing (200 themes → batches of 50)
  4. Real-time progress streaming via queue
  5. Theme-specific specialized agents

Run from CLI:
  python multi_agent.py "top signon issues in costco"
"""

import json
import re
import asyncio
from asyncio import Semaphore
from typing import TypedDict, List
from queue import Queue
import sys

import pandas as pd
from langchain_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

import config
from tools_and_prompts import get_filtering_options, filter_contextual_data


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

VOLUME_THRESHOLD = 200
MAX_CONCURRENT_REQUESTS = 10
REQUESTS_PER_MINUTE = 50
BATCH_SIZE = 50

SPECIALIST_MAP = {
    "payment":       "payment_specialist",
    "transaction":   "transaction_specialist",
    "signon":        "auth_specialist",
    "reward":        "rewards_specialist",
    "dispute/fraud": "dispute_specialist",
    "card":          "card_specialist",
    "other":         "general_specialist",
}

# Global progress queue (for Streamlit streaming)
progress_queue = Queue()


# ══════════════════════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════════════════════

class ThemeForAnalysis(TypedDict):
    theme_name: str
    theme_level: str
    call_count: int
    dominant_call_reason: str
    friction_drivers: list
    key_solutions: list
    specialist_type: str


class ThemeAnalysisResult(TypedDict):
    theme_name: str
    theme_level: str
    call_count: int
    digital_failure: str
    root_cause: str
    actionable_fix: str
    fix_owner: str
    fix_rationale: str
    urgency_score: int
    ease_score: int
    priority_score: float
    evidence: str


class AdaptiveState(TypedDict):
    user_query: str
    plan_decision: str              # "answer" | "clarify" | "execute"
    clarification_needed: str
    direct_answer: str
    filters_applied: dict
    raw_data: pd.DataFrame
    themes_for_analysis: list[ThemeForAnalysis]
    navigation_log: str
    analysis_results: list[ThemeAnalysisResult]
    prioritized_results: list[ThemeAnalysisResult]
    final_report: str


# ══════════════════════════════════════════════════════════════════════════════
# LLM
# ══════════════════════════════════════════════════════════════════════════════

llm = ChatVertexAI(model="gemini-1.5-pro", temperature=0)

tools = [get_filtering_options, filter_contextual_data]
tool_map = {t.name: t for t in tools}
llm_tools = llm.bind_tools(tools)


async def _call_async(system: str, user: str) -> str:
    resp = await llm.ainvoke([SystemMessage(content=system), HumanMessage(content=user)])
    return resp.content


def _parse_json(text: str) -> dict:
    cleaned = re.sub(r"```json\s*|\s*```", "", text).strip()
    return json.loads(cleaned)


def _emit_progress(stage: str, **kwargs):
    """Emit progress update to queue (for Streamlit)."""
    progress_queue.put({"stage": stage, **kwargs})


# ══════════════════════════════════════════════════════════════════════════════
# NODE 1 — PLANNER
# ══════════════════════════════════════════════════════════════════════════════

PLANNER_SYSTEM = """You are a query planner for a digital friction analysis system.

Your job: Decide if the query needs analysis or can be answered directly.

Output ONLY valid JSON:
{
  "decision": "answer" | "clarify" | "execute",
  "confidence": 0-100,
  "reasoning": "why you chose this",
  "response": "direct answer if decision=answer, clarification question if decision=clarify, empty if decision=execute"
}

RULES:
- decision="answer" if: general question about the system, asking for definitions, or simple factual query
  Examples: "what can you do?", "how does this work?", "what's digital friction?"

- decision="clarify" if: ambiguous product/theme, missing time period, unclear scope
  Examples: "analyze payment issues" (which product?), "recent problems" (how recent?)

- decision="execute" if: specific analysis request with clear scope
  Examples: "top signon issues in costco for Q4", "payment failures in walmart mobile app"

CONFIDENCE:
- 90-100: Very clear which path
- 70-89: Probably correct but some ambiguity
- Below 70: Use "clarify" to ask user
"""


async def planner_node(state: AdaptiveState) -> dict:
    """Decides whether to answer directly, clarify, or execute full analysis."""
    query = state["user_query"]
    
    _emit_progress("planner", message="Planning query routing...")
    
    response = await _call_async(system=PLANNER_SYSTEM, user=f"Query: {query}")
    
    try:
        plan = _parse_json(response)
        
        print(f"\n[Planner] Decision: {plan['decision']} (confidence: {plan['confidence']}%)")
        print(f"[Planner] Reasoning: {plan['reasoning']}")
        
        _emit_progress("planner", decision=plan["decision"], confidence=plan["confidence"])
        
        return {
            "plan_decision": plan["decision"],
            "clarification_needed": plan["response"] if plan["decision"] == "clarify" else "",
            "direct_answer": plan["response"] if plan["decision"] == "answer" else "",
        }
    
    except Exception as e:
        print(f"[Planner] Error: {e}, defaulting to execute")
        return {"plan_decision": "execute"}


def route_after_planner(state: AdaptiveState) -> str:
    """Routes to the appropriate node based on planner decision."""
    decision = state.get("plan_decision", "execute")
    
    if decision == "answer":
        return "direct_answer_node"
    elif decision == "clarify":
        return "clarifier_node"
    else:
        return "filter_node"


# ══════════════════════════════════════════════════════════════════════════════
# NODE 2 — DIRECT ANSWER / CLARIFIER
# ══════════════════════════════════════════════════════════════════════════════

async def direct_answer_node(state: AdaptiveState) -> dict:
    """Handles simple questions without running the full pipeline."""
    _emit_progress("direct_answer", message="Providing direct answer")
    return {"final_report": state["direct_answer"]}


async def clarifier_node(state: AdaptiveState) -> dict:
    """Returns clarification question to user."""
    _emit_progress("clarifier", message="Asking for clarification")
    return {"final_report": f"❓ **CLARIFICATION NEEDED:**\n\n{state['clarification_needed']}"}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3 — FILTER
# ══════════════════════════════════════════════════════════════════════════════

FILTER_SYSTEM = """You are a data filter agent. Call tools to retrieve customer call data.

Step 1: Call get_filtering_options to see valid values.
Step 2: Match user's words to exact values (fuzzy match).
Step 3: Call filter_contextual_data with matched values.
Step 4: Return the raw result from filter_contextual_data as-is."""


async def filter_node(state: AdaptiveState) -> dict:
    """Calls tools to get filtered data."""
    print(f"\n[Filter] Processing query: {state['user_query']}")
    _emit_progress("filter", message="Loading data...")
    
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
    
    data = pd.read_csv(config.DATA_FILE)
    
    print(f"[Filter] Dataset loaded: {len(data):,} total rows")
    _emit_progress("filter", message=f"Loaded {len(data):,} records")
    
    return {"raw_data": data, "filters_applied": {}}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 4 — HIERARCHY NAVIGATOR
# ══════════════════════════════════════════════════════════════════════════════

async def hierarchy_navigator(state: AdaptiveState) -> dict:
    """Adaptive hierarchy navigation with drill-down logic."""
    df = state["raw_data"]
    if df is None or len(df) == 0:
        return {"themes_for_analysis": [], "navigation_log": "No data to analyse."}
    
    print(f"\n[Navigator] Starting adaptive drill-down (threshold: {VOLUME_THRESHOLD})...")
    _emit_progress("navigator", message="Navigating theme hierarchy...")
    
    log = []
    final_themes = []
    
    # Level 1: Broad Theme
    broad = df.groupby("broad_theme").agg(
        volume=("call_id", "count"),
        dominant_call_reason=("call_reason", lambda x: x.mode()[0] if len(x) > 0 else "other"),
    ).reset_index()
    
    log.append(f"Level 1 (Broad): {len(broad)} themes")
    
    for _, row in broad.iterrows():
        theme = row["broad_theme"]
        vol = row["volume"]
        
        if vol <= VOLUME_THRESHOLD:
            theme_df = df[df["broad_theme"] == theme]
            final_themes.append(_extract_theme_data(theme_df, theme, "broad", row["dominant_call_reason"]))
            log.append(f"  ✓ '{theme}' ({vol}) → keep at BROAD level")
        else:
            log.append(f"  ↓ '{theme}' ({vol}) → drill to INTERMEDIATE")
            _drill_intermediate(df, theme, final_themes, log)
    
    navigation_log = "\n".join(log)
    print(f"[Navigator] Final: {len(final_themes)} themes selected for analysis")
    _emit_progress("navigator", themes_count=len(final_themes), message=navigation_log)
    
    return {"themes_for_analysis": final_themes, "navigation_log": navigation_log}


def _drill_intermediate(df: pd.DataFrame, broad_name: str, final_themes: list, log: list):
    subset = df[df["broad_theme"] == broad_name]
    inter = subset.groupby("intermediate_theme").agg(
        volume=("call_id", "count"),
        dominant_call_reason=("call_reason", lambda x: x.mode()[0] if len(x) > 0 else "other"),
    ).reset_index()
    
    for _, row in inter.iterrows():
        theme = row["intermediate_theme"]
        vol = row["volume"]
        
        if vol <= VOLUME_THRESHOLD:
            theme_df = subset[subset["intermediate_theme"] == theme]
            final_themes.append(_extract_theme_data(theme_df, theme, "intermediate", row["dominant_call_reason"]))
            log.append(f"    ✓ '{theme}' ({vol}) → keep at INTERMEDIATE")
        else:
            log.append(f"    ↓ '{theme}' ({vol}) → drill to GRANULAR")
            _drill_granular(subset, theme, final_themes, log)


def _drill_granular(df: pd.DataFrame, inter_name: str, final_themes: list, log: list):
    subset = df[df["intermediate_theme"] == inter_name]
    gran = subset.groupby("granular_theme").agg(
        volume=("call_id", "count"),
        dominant_call_reason=("call_reason", lambda x: x.mode()[0] if len(x) > 0 else "other"),
    ).reset_index()
    
    for _, row in gran.iterrows():
        theme = row["granular_theme"]
        vol = row["volume"]
        theme_df = subset[subset["granular_theme"] == theme]
        final_themes.append(_extract_theme_data(theme_df, theme, "granular", row["dominant_call_reason"]))
        log.append(f"      ✓ '{theme}' ({vol}) → GRANULAR level")


def _extract_theme_data(theme_df: pd.DataFrame, theme_name: str, level: str, call_reason: str) -> ThemeForAnalysis:
    return {
        "theme_name": theme_name,
        "theme_level": level,
        "call_count": len(theme_df),
        "dominant_call_reason": call_reason,
        "friction_drivers": list(theme_df["friction_driver_digital"].dropna().unique())[:15],
        "key_solutions": list(theme_df["key_solution"].dropna().unique())[:10],
        "specialist_type": SPECIALIST_MAP.get(call_reason, "general_specialist"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 5 — EXECUTOR (RATE-LIMITED BATCH PROCESSING)
# ══════════════════════════════════════════════════════════════════════════════

SPECIALIST_PROMPTS = {
    "payment_specialist": """You are a Payment Flow Specialist. Analyse payment-related friction.
Focus on: payment confirmation gaps, payment method failures, payment date mismatches, AutoPay setup confusion.
Urgency: 5=revenue-blocking, 4=high call driver, 3=UX unclear, 2=minor edge, 1=rare.
Ease: 5=UI label change, 4=UI component, 3=feature flag, 2=new feature, 1=backend integration.""",
    
    "auth_specialist": """You are an Authentication Specialist. Analyse sign-on/login friction.
Focus on: OTP delivery failures, password reset failures, biometric auth failures, account lockout.
Urgency: 5=locked out, 4=multiple attempts needed, 3=alternate method works, 2=inconvenience, 1=rare.
Ease: 5=messaging update, 4=retry/resend button, 3=timeout config, 2=fallback method, 1=provider replacement.""",
    
    "transaction_specialist": """You are a Transaction Clarity Specialist. Analyse transaction display issues.
Focus on: transaction labeling, categorization, linking, pending vs posted confusion.
Urgency: 5=fraud dispute calls, 4=high volume clarification, 3=confusion but resolvable, 2=cosmetic, 1=rare.
Ease: 5=relabel display, 4=link transactions, 3=drill-down feature, 2=normalization pipeline, 1=processor logic.""",
    
    "general_specialist": """You are a General Digital Experience Specialist.
Analyse any friction point. Default: urgency=3, ease=3 when unsure.""",
}

for k in ["rewards_specialist", "dispute_specialist", "card_specialist"]:
    SPECIALIST_PROMPTS[k] = SPECIALIST_PROMPTS["general_specialist"]

EXECUTOR_OUTPUT_FORMAT = """Output ONLY valid JSON:
{
  "digital_failure": "specific 1-sentence failure",
  "root_cause": "1 sentence why it fails",
  "actionable_fix": "1 sentence concrete fix",
  "fix_owner": "UI | Feature | Ops | Education",
  "fix_rationale": "1 sentence why this fix resolves root cause",
  "urgency_score": 1-5 integer,
  "ease_score": 1-5 integer,
  "evidence": "best verbatim item from friction_drivers"
}"""


async def execute_all_themes(state: AdaptiveState) -> dict:
    """Process all themes with rate limiting and batch progress."""
    themes = state["themes_for_analysis"]
    if not themes:
        return {"analysis_results": []}
    
    print(f"\n[Executor] Processing {len(themes)} themes with rate limiting...")
    _emit_progress("executor", message=f"Starting analysis of {len(themes)} themes...", total=len(themes))
    
    # Split into batches
    batches = [themes[i:i + BATCH_SIZE] for i in range(0, len(themes), BATCH_SIZE)]
    
    all_results = []
    
    for batch_num, batch in enumerate(batches, 1):
        print(f"\n📦 Batch {batch_num}/{len(batches)} ({len(batch)} themes)")
        _emit_progress("executor", batch=batch_num, total_batches=len(batches))
        
        batch_results = await _process_batch_with_rate_limit(batch, len(all_results), len(themes))
        all_results.extend(batch_results)
        
        print(f"   ✓ Batch {batch_num} complete. Total: {len(all_results)}/{len(themes)}")
        
        # Cool-down between batches
        if batch_num < len(batches):
            wait_time = max(1, 60 / (REQUESTS_PER_MINUTE / BATCH_SIZE))
            print(f"   ⏸ Cooling down {wait_time:.1f}s...")
            await asyncio.sleep(wait_time)
    
    return {"analysis_results": all_results}


async def _process_batch_with_rate_limit(themes: List[ThemeForAnalysis], completed_so_far: int, total: int) -> List[ThemeAnalysisResult]:
    """Process a single batch with semaphore-based rate limiting."""
    semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)
    
    async def _rate_limited_analyse(theme):
        async with semaphore:
            return await _analyse_single_theme_with_retry(theme)
    
    tasks = [_rate_limited_analyse(theme) for theme in themes]
    results = []
    
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        
        progress = (completed_so_far + len(results)) / total
        print(f"  ✓ [{completed_so_far + len(results)}/{total}] {result['theme_name'][:40]}")
        
        _emit_progress(
            "executor",
            theme=result['theme_name'],
            progress=progress,
            completed=completed_so_far + len(results),
            total=total,
            fix=result['actionable_fix'],
            owner=result['fix_owner'],
            urgency=result['urgency_score'],
            ease=result['ease_score']
        )
    
    return results


async def _analyse_single_theme_with_retry(theme: ThemeForAnalysis, max_retries=3) -> ThemeAnalysisResult:
    """Retry logic for rate limit errors."""
    for attempt in range(max_retries):
        try:
            return await _analyse_single_theme_async(theme)
        
        except Exception as e:
            error_msg = str(e).lower()
            
            if "rate limit" in error_msg or "429" in error_msg:
                backoff = 2 ** attempt
                print(f"  ⚠ Rate limit hit for '{theme['theme_name']}' - retry in {backoff}s")
                await asyncio.sleep(backoff)
            else:
                print(f"  ❌ Error analysing '{theme['theme_name']}': {e}")
                return _create_error_result(theme, str(e))
    
    return _create_error_result(theme, "Max retries exceeded")


async def _analyse_single_theme_async(theme: ThemeForAnalysis) -> ThemeAnalysisResult:
    """Core analysis logic for a single theme."""
    spec_type = theme["specialist_type"]
    system = SPECIALIST_PROMPTS[spec_type] + "\n\n" + EXECUTOR_OUTPUT_FORMAT
    
    user_prompt = (
        f"Theme: {theme['theme_name']}\n"
        f"Level: {theme['theme_level']}\n"
        f"Call Count: {theme['call_count']}\n"
        f"Call Reason: {theme['dominant_call_reason']}\n\n"
        f"Friction Drivers:\n" + "\n".join(f"  - {d}" for d in theme["friction_drivers"][:12]) + "\n\n"
        f"Key Solutions:\n" + "\n".join(f"  - {s}" for s in theme["key_solutions"][:8])
    )
    
    response = await _call_async(system=system, user=user_prompt)
    
    parsed = _parse_json(response)
    return {
        "theme_name": theme["theme_name"],
        "theme_level": theme["theme_level"],
        "call_count": theme["call_count"],
        "digital_failure": parsed.get("digital_failure", ""),
        "root_cause": parsed.get("root_cause", ""),
        "actionable_fix": parsed.get("actionable_fix", ""),
        "fix_owner": parsed.get("fix_owner", ""),
        "fix_rationale": parsed.get("fix_rationale", ""),
        "urgency_score": int(parsed.get("urgency_score", 3)),
        "ease_score": int(parsed.get("ease_score", 3)),
        "priority_score": 0.0,
        "evidence": parsed.get("evidence", ""),
    }


def _create_error_result(theme: ThemeForAnalysis, error: str) -> ThemeAnalysisResult:
    return {
        "theme_name": theme["theme_name"],
        "theme_level": theme["theme_level"],
        "call_count": theme["call_count"],
        "digital_failure": f"Analysis failed: {error}",
        "root_cause": "", "actionable_fix": "", "fix_owner": "",
        "fix_rationale": "", "urgency_score": 3, "ease_score": 3,
        "priority_score": 0.0, "evidence": "",
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 6 — PRIORITIZER
# ══════════════════════════════════════════════════════════════════════════════

async def prioritizer(state: AdaptiveState) -> dict:
    """Ranks results by urgency × ease × (volume/1000)."""
    results = state.get("analysis_results", [])
    if not results:
        return {"prioritized_results": []}
    
    print(f"\n[Prioritizer] Computing priority scores for {len(results)} themes...")
    _emit_progress("prioritizer", message="Prioritizing results...")
    
    for r in results:
        r["priority_score"] = r["urgency_score"] * r["ease_score"] * (r["call_count"] / 1000.0)
    
    sorted_results = sorted(results, key=lambda x: x["priority_score"], reverse=True)
    
    print("[Prioritizer] Top 3 by priority:")
    for i, r in enumerate(sorted_results[:3], 1):
        print(f"  {i}. {r['theme_name'][:50]} | priority={r['priority_score']:.1f}")
    
    _emit_progress("prioritizer", top_themes=[r['theme_name'] for r in sorted_results[:3]])
    
    return {"prioritized_results": sorted_results}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 7 — REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

REPORT_SYSTEM = """Generate a clear executive report in markdown.

Format:
---
## 🔍 Digital Friction Analysis
**Query:** {query}
**Themes analysed:** {n}
**Total call volume:** {total}

---
### #{rank} · {theme_name} ({level}) · {call_count} calls
**Priority Score:** {priority_score:.1f} (U:{urgency}/5 × E:{ease}/5)
**Digital Failure:** {digital_failure}
**Root Cause:** {root_cause}
**Actionable Fix [{fix_owner}]:** {actionable_fix}
*Rationale:* {fix_rationale}
**Evidence:** "{evidence}"

---

## 📊 Priority Matrix
- **Quick Wins** (U≥4, E≥4): [list]
- **Strategic Investments** (U≥4, E≤2): [list]

## ⚡ Top 3 Immediate Actions
1. [highest priority with owner]
2. [second]
3. [third]
"""


async def report_generator(state: AdaptiveState) -> dict:
    """Generates final markdown report."""
    results = state.get("prioritized_results", [])
    if not results:
        return {"final_report": "No themes analysed."}
    
    print(f"\n[Report] Generating final report for {len(results)} themes...")
    _emit_progress("report", message="Generating final report...")
    
    total_calls = sum(r["call_count"] for r in results)
    numbered = [{"rank": i + 1, **r} for i, r in enumerate(results)]
    
    user_prompt = (
        f"Query: {state['user_query']}\n"
        f"Themes: {len(results)}\n"
        f"Total calls: {total_calls:,}\n\n"
        f"Prioritized themes:\n{json.dumps(numbered, indent=2)}"
    )
    
    report = await _call_async(system=REPORT_SYSTEM, user=user_prompt)
    
    print("[Report] Complete.")
    _emit_progress("report", message="Report complete!")
    
    return {"final_report": report}


# ══════════════════════════════════════════════════════════════════════════════
# BUILD GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def build_graph():
    graph = StateGraph(AdaptiveState)
    
    # Add all nodes
    graph.add_node("planner_node", planner_node)
    graph.add_node("direct_answer_node", direct_answer_node)
    graph.add_node("clarifier_node", clarifier_node)
    graph.add_node("filter_node", filter_node)
    graph.add_node("hierarchy_navigator", hierarchy_navigator)
    graph.add_node("execute_all_themes", execute_all_themes)
    graph.add_node("prioritizer", prioritizer)
    graph.add_node("report_generator", report_generator)
    
    # Routing
    graph.add_edge(START, "planner_node")
    graph.add_conditional_edges(
        "planner_node",
        route_after_planner,
        {
            "direct_answer_node": "direct_answer_node",
            "clarifier_node": "clarifier_node",
            "filter_node": "filter_node",
        }
    )
    
    # Direct paths to END
    graph.add_edge("direct_answer_node", END)
    graph.add_edge("clarifier_node", END)
    
    # Analysis pipeline
    graph.add_edge("filter_node", "hierarchy_navigator")
    graph.add_edge("hierarchy_navigator", "execute_all_themes")
    graph.add_edge("execute_all_themes", "prioritizer")
    graph.add_edge("prioritizer", "report_generator")
    graph.add_edge("report_generator", END)
    
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


graph = build_graph()


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════════

async def run_async(query: str, thread_id: str = "default") -> str:
    """Run the graph asynchronously."""
    initial_state: AdaptiveState = {
        "user_query": query,
        "plan_decision": "",
        "clarification_needed": "",
        "direct_answer": "",
        "filters_applied": {},
        "raw_data": None,
        "themes_for_analysis": [],
        "navigation_log": "",
        "analysis_results": [],
        "prioritized_results": [],
        "final_report": "",
    }
    
    result = await graph.ainvoke(initial_state, config={"configurable": {"thread_id": thread_id}})
    return result["final_report"]


def run(query: str, thread_id: str = "default") -> str:
    """Synchronous wrapper for CLI usage."""
    return asyncio.run(run_async(query, thread_id))


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python multi_agent.py 'your query here'")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    print(f"Running query: {query}\n")
    
    report = run(query)
    print("\n" + "=" * 80)
    print(report)
