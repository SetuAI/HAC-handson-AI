"""
agents/supervisor.py — Supervisor Agent (The Orchestrator)

The supervisor runs after EVERY agent completes.
It checks the current state and decides who runs next.

Execution order it enforces:
    1. market_data_agent      → get price & company data
    2. risk_analyst_agent     → calculate risk metrics
    3. fundamental_analyst    → calculate valuation metrics
    4. report_writer_agent    → synthesize final report
    5. END                    → pipeline complete
"""

import logging
from typing import Literal

from core.config import settings, FinancialAnalysisState, AgentNames

logger = logging.getLogger(__name__)


# =============================================================================
# SUPERVISOR NODE
# =============================================================================

def supervisor_node(state: FinancialAnalysisState) -> dict:
    """
    Reads the current state and decides which agent runs next.

    This is pure routing logic — no LLM call needed here.
    We just check which fields are empty and route accordingly.

    Args:
        state: the full shared state (the "baton")

    Returns:
        dict with next_agent set to whoever should run next
    """
    logger.info("[SUPERVISOR] Evaluating pipeline progress...")

    # Check what has been filled in so far
    has_market_data      = bool(state.get("market_data"))
    has_risk_metrics     = bool(state.get("risk_metrics"))
    has_fundamental_data = bool(state.get("fundamental_data"))
    has_report           = bool(state.get("final_report"))

    logger.info(
        f"[SUPERVISOR] market={has_market_data} | "
        f"risk={has_risk_metrics} | "
        f"fundamental={has_fundamental_data} | "
        f"report={has_report}"
    )

    # Waterfall routing — first unfilled stage wins
    if not has_market_data:
        next_agent = AgentNames.MARKET_DATA_AGENT
        msg        = "No market data yet → routing to Market Data Agent"

    elif not has_risk_metrics:
        next_agent = AgentNames.RISK_ANALYST_AGENT
        msg        = "No risk metrics yet → routing to Risk Analyst"

    elif not has_fundamental_data:
        next_agent = AgentNames.FUNDAMENTAL_ANALYST
        msg        = "No fundamental data yet → routing to Fundamental Analyst"

    elif not has_report:
        next_agent = AgentNames.REPORT_WRITER
        msg        = "All data ready → routing to Report Writer"

    else:
        next_agent = AgentNames.END
        msg        = "Report done → pipeline complete"

    logger.info(f"[SUPERVISOR] {msg}")

    return {
        "next_agent"  : next_agent,
        "agent_steps" : [f"SUPERVISOR → {next_agent}: {msg}"],
    }


# =============================================================================
# CONDITIONAL ROUTER
# =============================================================================

def route_to_next_agent(
    state: FinancialAnalysisState,
) -> Literal[
    "market_data_agent",
    "risk_analyst_agent",
    "fundamental_analyst_agent",
    "report_writer_agent",
    "__end__",
]:
    """
    Called by LangGraph after every node to decide which edge to follow.

    This is NOT a node — it's a routing function used by add_conditional_edges().
    It simply reads next_agent from state and returns the matching node name.

    LangGraph requires the return value to exactly match one of the
    keys in the mapping dict we define in pipeline.py.

    Returns:
        string name of the next node, or "__end__" to stop the graph
    """
    next_agent = state.get("next_agent", AgentNames.END)

    # AgentNames.END = "END" but LangGraph's termination constant is "__end__"
    if next_agent == AgentNames.END:
        return "__end__"

    return next_agent