```python
"""
tools.py + agent_prompts.py
----------------------------
Properly described tools and agent system prompts for the
Customer Pain Point natural language query system.

Natural language target:
  "Show top issues in Costco for sign on"
  → Router Agent calls get_filtering_options to validate filters
  → Router Agent calls filter_contextual_data with {product: costco, call_theme: signon}
  → Passes results to Digital Friction Specialist Agent
  → Specialist synthesises insight report
"""

import pandas as pd
from typing import List, Optional
from langchain_core.tools import tool
import config  # your existing config module


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 1: GET FILTERING OPTIONS
# ══════════════════════════════════════════════════════════════════════════════

@tool
def get_filtering_options() -> dict:
    """
    Returns all valid filter values available in the customer call dataset.

    Use this tool FIRST — before calling filter_contextual_data — to discover
    what valid product names, call themes, and other filter values exist in the data.
    This prevents filtering on values that don't exist (e.g. 'Costco' vs 'Costco Anywhere Visa').

    Returns a dictionary where each key is a filterable column name and each
    value is a list of valid options for that column. Example:
    {
        "product":     ["Costco Anywhere Visa", "Visa Signature", "World Mastercard", ...],
        "call_theme":  ["signon", "payment", "transaction", "reward", "dispute/fraud", ...],
        "domain":      ["Authentication", "Transaction", "Payments", ...]
    }

    Always call this when the user mentions a product or theme by name, so you can
    match their natural language input to the exact values in the dataset.
    """
    data = pd.read_csv(config.DATA_FILE)
    return {col: sorted(list(data[col].dropna().unique())) for col in config.FILTERING_COLUMNS}


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 2: FILTER CONTEXTUAL DATA
# ══════════════════════════════════════════════════════════════════════════════

@tool
def filter_contextual_data(
    product: Optional[List[str]] = None,
    call_theme: Optional[List[str]] = None,
) -> dict:
    """
    Filters the customer call dataset and returns the top 10 pain point clusters
    ranked by call volume for the given product and/or call theme.

    Use this tool to answer questions like:
      - "What are the top issues for Costco customers?"
      - "What sign-on problems are customers calling about?"
      - "Show me the top friction areas for Costco customers with signon issues"

    Each result row represents a granular pain point theme and includes:
      - call_counts:            number of calls in this theme (higher = more impactful)
      - granular_theme:         the specific cluster label describing the pain point
      - friction_driver_digital: list of digital touchpoints causing the calls
      - key_solution:           list of recommended solutions identified from these calls

    Results are sorted by call_counts descending so the highest-volume problems appear first.

    Args:
        product:    List of product names to filter by. Must be exact values from
                    get_filtering_options()['product']. Examples: ["Costco Anywhere Visa"].
                    Pass None or omit to include all products.

        call_theme: List of call theme/reason values to filter by. Must be exact values from
                    get_filtering_options()['call_theme']. Examples: ["signon"], ["payment", "transaction"].
                    Pass None or omit to include all call themes.

    Returns:
        Dictionary with up to 10 entries. Each key is a row index, each value contains:
        {
            "granular_theme":         "OTP Not Received on Mobile",
            "call_counts":            847,
            "friction_driver_digital": ["OTP delivery failure", "No retry prompt shown", ...],
            "key_solution":           ["Add resend OTP button", "Show delivery status", ...]
        }

    Important: Always call get_filtering_options() first to get valid filter values
    before calling this tool, especially when user input is informal (e.g. "costco"
    should be matched to "Costco Anywhere Visa").
    """
    data = pd.read_csv(config.DATA_FILE)

    # Apply filters only if provided and non-empty
    if product:
        data = data[data["product"].isin(product)]
    if call_theme:
        data = data[data["call_theme"].isin(call_theme)]

    # Aggregate to granular theme level, ranked by call volume
    data = (
        data.groupby("granular_theme")
        .agg(
            call_counts            = ("call_id", "count"),
            friction_driver_digital = ("friction_driver_digital", list),
            key_solution           = ("key_solution", list),
        )
        .reset_index()
        .sort_values("call_counts", ascending=False)
    )

    # Return top 10 as dict for LLM consumption
    top_n = min(len(data), 10)
    return data.iloc[:top_n].to_dict("index")


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 1 SYSTEM PROMPT — ROUTER / QUERY AGENT
# ══════════════════════════════════════════════════════════════════════════════

ROUTER_AGENT_PROMPT = """You are a Customer Pain Point Query Agent for a digital banking team.

You have access to a dataset of {total_calls} customer calls. Each call was made after a \
customer visited the digital banking platform and still needed to call for help.

Your job is to:
1. Understand what the user is asking about (which team, product, call theme, issue type)
2. Use get_filtering_options to discover valid filter values in the data
3. Match the user's natural language to exact filter values (e.g. "costco" → "Costco Anywhere Visa")
4. Use filter_contextual_data to retrieve the ranked pain point clusters
5. Pass the structured results to the Digital Friction Specialist for insight synthesis

WORKFLOW — always follow this order:
  Step 1: Call get_filtering_options() to see available products and themes
  Step 2: Map user's words to exact values (fuzzy match if needed)
  Step 3: Call filter_contextual_data() with the matched filter values
  Step 4: Hand the results to the Digital Friction Specialist with context

FILTER MATCHING EXAMPLES:
  User says "costco"          → product: ["Costco Anywhere Visa"] (or all Costco variants)
  User says "sign on / login" → call_theme: ["signon"]
  User says "payment issues"  → call_theme: ["payment", "transaction"]
  User says "fraud"           → call_theme: ["dispute/fraud"]
  User says "top issues"      → no theme filter, show all

If you are unsure how to map the user's words to a filter value, show the user
the available options and ask them to clarify — do not guess incorrectly.

If the user asks a question that doesn't require filtering (e.g. "what teams exist?"),
answer directly from get_filtering_options() without calling filter_contextual_data.

Always be clear about:
  - How many total calls matched their filters
  - What filters were applied
  - That the results are ranked by call volume (highest impact first)
"""

# ══════════════════════════════════════════════════════════════════════════════
# AGENT 2 SYSTEM PROMPT — DIGITAL FRICTION SPECIALIST
# ══════════════════════════════════════════════════════════════════════════════

DIGITAL_FRICTION_SPECIALIST_PROMPT = """You are a Digital Friction Specialist for a digital banking team.

You receive pre-filtered, ranked customer pain point data and your job is to synthesise it into
clear, specific, actionable insights for product and operations teams.

You will receive a dictionary of the top pain point clusters for a specific product/theme combination.
Each cluster contains:
  - granular_theme:          the name of the specific pain point cluster
  - call_counts:             how many customers called about this issue
  - friction_driver_digital: list of digital touchpoints/failures that caused the calls
  - key_solution:            list of recommended solutions

YOUR OUTPUT FORMAT — always structure your response as follows:

---
## Digital Friction Report
**Filters applied:** {product} | {call_theme}
**Total calls in view:** {total}
**Top {n} issues by call volume**

---
### #{rank} — {granular_theme}
**Volume:** {call_counts} calls ({pct}% of filtered total)
**What's happening:** [1-2 sentence specific description of the exact digital failure]
**Digital friction point:** [what in the digital experience is causing this]
**Recommended fix:** [the most actionable recommendation from key_solution]
**Fix type:** [UI / Feature / Ops / Education]

---
[repeat for each issue]

## Summary
[2-3 sentences on the overall pattern across these issues — is this a UI problem?
a notification problem? a labelling problem? Give the team one clear theme to act on]
---

RULES:
- Be SPECIFIC — not "transaction clarity issues" but "cash advance fee displayed as
  'cash withdrawal' — customer cannot identify the $20 as a fee"
- The 'What's happening' must describe something a product team can fix tomorrow
- Always include call volume and % — this is what prioritises the work
- Fix type must be one of: UI, Feature, Ops, Education
- If friction_driver_digital contains multiple items, identify the PRIMARY one
- If key_solution contains multiple items, pick the most impactful/actionable one
- End with the Summary — the team needs to know if these are all the same root cause
  or genuinely separate problems
"""


# ══════════════════════════════════════════════════════════════════════════════
# HOW TO WIRE THE TWO AGENTS TOGETHER
# ══════════════════════════════════════════════════════════════════════════════
"""
OPTION A: Single agent with both tools + specialist prompt injected at output stage
  → Simpler, one LangGraph node
  → Router agent collects data, then switches to specialist persona for synthesis

OPTION B: True two-agent pipeline (recommended for your architecture)
  → Router Agent: has get_filtering_options + filter_contextual_data tools
  → Specialist Agent: receives structured data, no tools, pure synthesis
  → Router passes results to Specialist via state

See wiring example below:
"""

from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

# ── Router Agent — has tools, handles filter discovery + data retrieval ───────
router_agent = create_react_agent(
    model=llm,
    tools=[get_filtering_options, filter_contextual_data],
    state_modifier=ROUTER_AGENT_PROMPT,
)

# ── Specialist Agent — no tools, pure synthesis from structured data ──────────
specialist_agent = create_react_agent(
    model=llm,
    tools=[],  # no tools — receives data in the message, synthesises insight
    state_modifier=DIGITAL_FRICTION_SPECIALIST_PROMPT,
)


# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE: Two-step invocation
# ══════════════════════════════════════════════════════════════════════════════

def run_pain_point_query(user_query: str) -> str:
    """
    Full two-agent pipeline for a natural language pain point query.

    Example:
        result = run_pain_point_query(
            "can you show which are top issues in costco for sign on"
        )
    """
    # Step 1: Router agent retrieves and filters data
    router_result = router_agent.invoke({
        "messages": [{"role": "user", "content": user_query}]
    })
    router_output = router_result["messages"][-1].content

    # Step 2: Specialist agent synthesises insight from router's output
    specialist_result = specialist_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Original user question: '{user_query}'\n\n"
                    f"Filtered pain point data retrieved:\n\n"
                    f"{router_output}\n\n"
                    f"Please synthesise this into a Digital Friction Report."
                )
            }
        ]
    })

    return specialist_result["messages"][-1].content


# ══════════════════════════════════════════════════════════════════════════════
# TEST IT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    queries = [
        "can you show which are top issues in costco for sign on",
        "what are the biggest payment problems across all products",
        "top 5 issues for authentication team",
        "show me the most common reasons costco customers are calling",
    ]

    for q in queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {q}")
        print('='*60)
        print(run_pain_point_query(q))
