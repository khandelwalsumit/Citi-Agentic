"""
app.py
------
Streamlit UI for the multi-agent digital friction analyzer.

Run:
  streamlit run app.py
"""

import streamlit as st
import asyncio
from threading import Thread
import time

from multi_agent import run_async, progress_queue


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: RUN GRAPH IN BACKGROUND THREAD
# ══════════════════════════════════════════════════════════════════════════════

def run_graph_in_background(query: str):
    """Run async graph in a separate thread."""
    def _run():
        asyncio.run(run_async(query))
    
    thread = Thread(target=_run, daemon=True)
    thread.start()
    return thread


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Digital Friction Analyzer", page_icon="🔍", layout="wide")

st.title("🔍 Digital Friction Analyzer")
st.markdown("**AI-powered analysis of customer friction points in digital channels**")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This tool analyzes customer call data to identify digital friction points.
    
    **Features:**
    - 🧠 Smart query routing (answer/clarify/execute)
    - 🗂 Adaptive theme hierarchy navigation
    - ⚡ Rate-limited batch processing
    - 🎯 Specialized agents per call type
    - 📊 Priority-ranked actionable fixes
    """)
    
    st.divider()
    
    st.header("📋 Example Queries")
    examples = [
        "top signon issues in costco",
        "payment failures in walmart mobile app",
        "what is digital friction?",
        "analyze recent problems"
    ]
    
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state.query = ex

# Main input
query = st.text_input(
    "Enter your query:",
    value=st.session_state.get("query", "top signon issues in costco"),
    placeholder="e.g., 'payment issues in walmart' or 'what can you do?'"
)

col1, col2 = st.columns([1, 5])
with col1:
    analyze_btn = st.button("🚀 Analyze", type="primary")
with col2:
    if st.button("🗑 Clear"):
        st.session_state.clear()
        st.rerun()

if analyze_btn and query:
    # Clear previous results
    st.session_state.results = []
    
    # Create UI placeholders
    status_placeholder = st.empty()
    progress_container = st.container()
    results_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0, text="Starting analysis...")
    
    # Start graph in background
    thread = run_graph_in_background(query)
    
    # Poll progress queue
    theme_count = 0
    completed_themes = []
    
    while thread.is_alive() or not progress_queue.empty():
        try:
            update = progress_queue.get(timeout=0.1)
            
            # ── Planner Stage ─────────────────────────────────────────────────
            if update["stage"] == "planner":
                if "decision" in update:
                    status_placeholder.info(
                        f"🧠 **Planning:** Routing to **{update['decision']}** "
                        f"(confidence: {update['confidence']}%)"
                    )
            
            # ── Direct Answer ─────────────────────────────────────────────────
            elif update["stage"] == "direct_answer":
                status_placeholder.success("✅ Direct answer provided")
            
            # ── Clarifier ─────────────────────────────────────────────────────
            elif update["stage"] == "clarifier":
                status_placeholder.warning("❓ Clarification needed")
            
            # ── Filter Stage ──────────────────────────────────────────────────
            elif update["stage"] == "filter":
                status_placeholder.info(f"📊 **Loading data:** {update.get('message', '')}")
            
            # ── Navigator Stage ───────────────────────────────────────────────
            elif update["stage"] == "navigator":
                if "themes_count" in update:
                    theme_count = update["themes_count"]
                    status_placeholder.info(
                        f"🗂 **Hierarchy navigation complete:** {theme_count} themes identified"
                    )
                    with st.expander("📋 Navigation Log", expanded=False):
                        st.text(update.get("message", ""))
            
            # ── Executor Stage ────────────────────────────────────────────────
            elif update["stage"] == "executor":
                if "progress" in update:
                    progress_pct = update["progress"]
                    completed = update.get("completed", 0)
                    total = update.get("total", theme_count)
                    
                    progress_bar.progress(
                        progress_pct,
                        text=f"🔬 Analyzing themes: {completed}/{total}"
                    )
                    
                    status_placeholder.info(f"🔬 **Analyzing:** {update.get('theme', '')[:60]}")
                    
                    # Display result as it arrives
                    if update.get("theme"):
                        completed_themes.append({
                            "theme": update["theme"],
                            "fix": update.get("fix", ""),
                            "owner": update.get("owner", ""),
                            "urgency": update.get("urgency", 3),
                            "ease": update.get("ease", 3),
                        })
                        
                        with results_container:
                            st.markdown(f"### ✓ {update['theme']}")
                            
                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                st.markdown(f"**Fix:** {update['fix'][:200]}")
                            with col2:
                                st.metric("Urgency", f"{update['urgency']}/5")
                            with col3:
                                st.metric("Ease", f"{update['ease']}/5")
                            
                            st.caption(f"Owner: {update['owner']}")
                            st.divider()
            
            # ── Prioritizer Stage ─────────────────────────────────────────────
            elif update["stage"] == "prioritizer":
                status_placeholder.info("📊 **Prioritizing results...**")
                if "top_themes" in update:
                    with st.expander("🏆 Top Themes", expanded=False):
                        for i, theme in enumerate(update["top_themes"], 1):
                            st.write(f"{i}. {theme}")
            
            # ── Report Stage ──────────────────────────────────────────────────
            elif update["stage"] == "report":
                status_placeholder.success("✅ **Generating final report...**")
        
        except:
            time.sleep(0.1)
            continue
    
    # Final report
    progress_bar.progress(1.0, text="Analysis complete!")
    status_placeholder.success("🎉 **Analysis Complete!**")
    
    # Get final state (this is a hack - ideally you'd store the result)
    # For now, we'll just show the streamed results
    st.divider()
    st.markdown("## 📊 Final Analysis")
    st.markdown(f"**Total themes analyzed:** {len(completed_themes)}")
    
    if completed_themes:
        st.markdown("### 🏆 Top Findings")
        # Sort by urgency × ease
        sorted_themes = sorted(
            completed_themes,
            key=lambda x: x["urgency"] * x["ease"],
            reverse=True
        )
        
        for i, theme in enumerate(sorted_themes[:5], 1):
            with st.container():
                st.markdown(f"**{i}. {theme['theme']}**")
                st.markdown(f"_{theme['fix']}_")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Owner", theme['owner'])
                with col2:
                    st.metric("Urgency", f"{theme['urgency']}/5")
                with col3:
                    st.metric("Ease", f"{theme['ease']}/5")
                
                st.divider()

# Footer
st.divider()
st.caption("Powered by LangGraph + Gemini | Multi-Agent Adaptive System")
