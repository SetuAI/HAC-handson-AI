"""
agents/risk_analyst_agent.py — Risk Analyst Specialist Agent

Second agent in the pipeline.
Calls calculate_risk_metrics, then uses the LLM to INTERPRET the numbers.

Key difference from Market Data Agent:
    - Market Data Agent just fetches and stores raw data
    - Risk Analyst fetches AND interprets — the LLM adds narrative value
    - Also injects market_data context from the previous agent's output
"""

import json
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from core.config import settings, FinancialAnalysisState
from tools.financial_tools import calculate_risk_metrics, ALL_RISK_TOOLS

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a Senior Quantitative Risk Analyst at a top-tier investment bank.

You have 15 years of experience analysing equity risk for institutional portfolios.

STEPS:
1. Call calculate_risk_metrics(ticker) to get the quantitative data
2. Interpret what those numbers MEAN for an investor
3. Give exactly 4-5 bullet-point insights

INTERPRETATION BENCHMARKS:
Volatility > 40%  → HIGH risk, aggressive investors only
Volatility 20-40% → MEDIUM risk, growth investors
Volatility < 20%  → LOWER risk, conservative investors
Beta > 1.5        → highly sensitive to market downturns
Beta 0.8-1.2      → moves with the market
Beta < 0.7        → defensive, good for hedging
Sharpe > 2.0      → excellent risk-adjusted returns
Sharpe 1.0-2.0    → acceptable
Sharpe < 1.0      → poor returns for the risk taken
Max Drawdown > 50%→ extreme historical loss, be cautious

FORMAT FOR EACH INSIGHT:
- Cite the actual number
- Explain what it means in plain English
- State what an investor should consider

Example: "Beta of 1.8 means this stock moves 80% more than the market.
In a 10% market correction, expect roughly an 18% drop."
"""


def risk_analyst_agent_node(state: FinancialAnalysisState) -> dict:
    """
    Calculates and interprets risk metrics for the given ticker.

    Reads market_data from state to inject context into the prompt.
    Writes risk_metrics and risk_insights back into state.

    Args:
        state: shared state — reads 'ticker' and 'market_data'

    Returns:
        partial state update with 'risk_metrics' and 'risk_insights'
    """
    ticker       = state.get("ticker", "UNKNOWN").upper()
    company_name = state.get("company_name", ticker)
    market_data  = state.get("market_data", {})

    logger.info(f"[RISK ANALYST] Starting for {company_name} ({ticker})")

    # --- Initialise LLM ---
    llm = ChatOpenAI(
        model       = settings.llm_model,
        temperature = settings.analyst_temperature,
        api_key     = settings.openai_api_key,
    ).bind_tools(ALL_RISK_TOOLS)

    # --- Inject context from the previous agent ---
    # This is the key MAS pattern: agents share context via shared state.
    # The Risk Analyst knows the current price and market cap because
    # the Market Data Agent already stored it in state["market_data"].
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Analyse the risk profile for {company_name} ({ticker}).\n\n"
                f"Context from Market Data Agent:\n"
                f"  Current price : ${market_data.get('current_price', 'N/A')}\n"
                f"  Market cap    : ${market_data.get('market_cap', 0):,.0f}\n"
                f"  Sector        : {market_data.get('sector', 'Unknown')}\n\n"
                f"Call calculate_risk_metrics to get the data, then provide your insights."
            )
        ),
    ]

    risk_metrics   = {}
    max_iterations = 3   # this agent needs at most 2 turns: tool call + interpret

    # --- ReAct loop ---
    for i in range(max_iterations):
        logger.info(f"[RISK ANALYST] LLM iteration {i + 1}")
        response = llm.invoke(messages)

        if not response.tool_calls:
            logger.info("[RISK ANALYST] LLM finished — no more tool calls")
            break

        messages.append(response)

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            logger.info(f"[RISK ANALYST] Calling {tool_name}({tool_args})")

            if tool_name == "calculate_risk_metrics":
                tool_result  = calculate_risk_metrics.invoke(tool_args)
                risk_metrics = tool_result   # capture for state update
            else:
                tool_result = {"error": f"Unknown tool: {tool_name}"}

            messages.append(
                ToolMessage(
                    content      = json.dumps(tool_result, default=str),
                    tool_call_id = tool_call["id"],
                )
            )

    # --- Extract the LLM's text interpretation ---
    # Walk backwards through messages to find the last text response
    analysis_text = ""
    for msg in reversed(messages):
        if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content:
            analysis_text = msg.content
            break

    # --- Parse bullet points into a list ---
    # Split on newlines, strip bullet characters, filter short lines
    insights = []
    if analysis_text:
        for line in analysis_text.strip().split("\n"):
            cleaned = line.strip().lstrip("•-*123456789. ")
            if len(cleaned) > 20:
                insights.append(cleaned)

    # Fallback: if parsing produced nothing, store raw text as one insight
    if not insights and analysis_text:
        insights = [analysis_text]

    logger.info(
        f"[RISK ANALYST] Done. "
        f"insights={len(insights)} | "
        f"risk_level={risk_metrics.get('risk_level', 'N/A')} | "
        f"vol={risk_metrics.get('annualized_volatility_pct', 'N/A')}%"
    )

    return {
        "risk_metrics"  : risk_metrics,
        "risk_insights" : insights,       # operator.add appends to existing list
        "agent_steps"   : [
            f"RISK_ANALYST: vol={risk_metrics.get('annualized_volatility_pct', 'N/A')}% | "
            f"beta={risk_metrics.get('beta', 'N/A')} | "
            f"sharpe={risk_metrics.get('sharpe_ratio', 'N/A')} | "
            f"risk={risk_metrics.get('risk_level', 'N/A')}"
        ],
    }