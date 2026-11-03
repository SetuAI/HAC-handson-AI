"""
agents/fundamental_analyst_agent.py — Fundamental Analyst Specialist Agent

Third agent in the pipeline.
Fetches valuation and financial health metrics, then interprets them.

Demonstrates a key MAS pattern:
    This agent injects context from BOTH previous agents —
    market price (from Market Data Agent) and risk level
    (from Risk Analyst Agent) — showing how agents build on
    each other's work through shared state.
"""

import json
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from core.config import settings, FinancialAnalysisState
from tools.financial_tools import fetch_fundamental_data, ALL_FUNDAMENTAL_TOOLS

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a Senior Fundamental Analyst at a value-oriented hedge fund.

You specialise in financial statement analysis and equity valuation
across all major sectors.

STEPS:
1. Call fetch_fundamental_data(ticker) to get the financial metrics
2. Assess VALUATION — is the stock cheap, fair value, or expensive?
3. Assess FINANCIAL HEALTH — is the business strong or fragile?
4. Give exactly 4-6 bullet-point insights

VALUATION BENCHMARKS:
P/E < 15        → potentially undervalued (check why growth may be slowing)
P/E 15-25       → fair value for moderate-growth companies
P/E 25-40       → growth premium, only justified if revenue growth > 15%
P/E > 40        → expensive, requires exceptional growth to justify
PEG < 1.0       → stock may be undervalued relative to its growth
PEG > 2.0       → growth is generously priced in, use caution

FINANCIAL HEALTH BENCHMARKS:
Net Margin > 20%    → excellent profitability, pricing power present
Net Margin 10-20%   → good profitability
Net Margin < 5%     → competitive pressure or structural issues
Debt/Equity < 0.5   → conservative, strong balance sheet
Debt/Equity 0.5-1.5 → moderate leverage, manageable
Debt/Equity > 2.0   → high leverage, vulnerable to rising interest rates
Free Cash Flow > 0  → generating real cash, not just accounting profit
ROE > 15%           → management using capital efficiently

FORMAT FOR EACH INSIGHT:
Cover in order: valuation, profitability, growth, financial health,
then optionally cash flow or dividends. Always cite the actual number.
"""


def fundamental_analyst_agent_node(state: FinancialAnalysisState) -> dict:
    """
    Fetches and interprets fundamental valuation and financial health metrics.

    Reads market_data AND risk_metrics from state to build a richer context
    prompt — this agent has the most context of any specialist agent.

    Args:
        state: shared state — reads ticker, market_data, risk_metrics

    Returns:
        partial state update with 'fundamental_data' and 'fundamental_insights'
    """
    ticker       = state.get("ticker", "UNKNOWN").upper()
    company_name = state.get("company_name", ticker)
    market_data  = state.get("market_data", {})
    risk_metrics = state.get("risk_metrics", {})

    logger.info(f"[FUNDAMENTAL ANALYST] Starting for {company_name} ({ticker})")

    # --- Initialise LLM ---
    llm = ChatOpenAI(
        model       = settings.llm_model,
        temperature = settings.analyst_temperature,
        api_key     = settings.openai_api_key,
    ).bind_tools(ALL_FUNDAMENTAL_TOOLS)

    # --- Inject context from BOTH previous agents ---
    # This agent has the richest context of any specialist:
    #   - sector/industry     from Market Data Agent
    #   - price, market cap   from Market Data Agent
    #   - risk level, beta    from Risk Analyst Agent
    # All of this helps the LLM calibrate its valuation benchmarks correctly.
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Perform fundamental analysis for {company_name} ({ticker}).\n\n"
                f"Context from previous agents:\n"
                f"  Sector        : {market_data.get('sector', 'Unknown')}\n"
                f"  Industry      : {market_data.get('industry', 'Unknown')}\n"
                f"  Current price : ${market_data.get('current_price', 'N/A')}\n"
                f"  Market cap    : ${market_data.get('market_cap', 0):,.0f}\n"
                f"  52-week range : ${market_data.get('week_52_low', 'N/A')} "
                f"- ${market_data.get('week_52_high', 'N/A')}\n"
                f"  Risk level    : {risk_metrics.get('risk_level', 'Unknown')}\n"
                f"  Beta          : {risk_metrics.get('beta', 'N/A')}\n\n"
                f"Call fetch_fundamental_data to get the metrics, "
                f"then provide your analysis."
            )
        ),
    ]

    fundamental_data = {}
    max_iterations   = 3

    # --- ReAct loop ---
    for i in range(max_iterations):
        logger.info(f"[FUNDAMENTAL ANALYST] LLM iteration {i + 1}")
        response = llm.invoke(messages)

        if not response.tool_calls:
            logger.info("[FUNDAMENTAL ANALYST] LLM finished — no more tool calls")
            break

        messages.append(response)

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            logger.info(f"[FUNDAMENTAL ANALYST] Calling {tool_name}({tool_args})")

            if tool_name == "fetch_fundamental_data":
                tool_result      = fetch_fundamental_data.invoke(tool_args)
                fundamental_data = tool_result
            else:
                tool_result = {"error": f"Unknown tool: {tool_name}"}

            messages.append(
                ToolMessage(
                    content      = json.dumps(tool_result, default=str),
                    tool_call_id = tool_call["id"],
                )
            )

    # --- Extract the LLM's text interpretation ---
    analysis_text = ""
    for msg in reversed(messages):
        if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content:
            analysis_text = msg.content
            break

    # --- Parse bullet points into a list ---
    insights = []
    if analysis_text:
        for line in analysis_text.strip().split("\n"):
            cleaned = line.strip().lstrip("•-*123456789. ")
            if len(cleaned) > 20:
                insights.append(cleaned)

    if not insights and analysis_text:
        insights = [analysis_text]

    logger.info(
        f"[FUNDAMENTAL ANALYST] Done. "
        f"insights={len(insights)} | "
        f"PE={fundamental_data.get('pe_ratio', 'N/A')} | "
        f"net_margin={fundamental_data.get('net_margin_pct', 'N/A')}% | "
        f"D/E={fundamental_data.get('debt_to_equity', 'N/A')}"
    )

    return {
        "fundamental_data"     : fundamental_data,
        "fundamental_insights" : insights,    # operator.add appends to existing list
        "agent_steps"          : [
            f"FUNDAMENTAL_ANALYST: PE={fundamental_data.get('pe_ratio', 'N/A')} | "
            f"EPS={fundamental_data.get('eps_ttm', 'N/A')} | "
            f"net_margin={fundamental_data.get('net_margin_pct', 'N/A')}% | "
            f"D/E={fundamental_data.get('debt_to_equity', 'N/A')}"
        ],
    }