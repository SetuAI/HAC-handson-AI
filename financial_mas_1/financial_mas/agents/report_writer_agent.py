"""
agents/report_writer_agent.py — Report Writer Specialist Agent

Final agent in the pipeline.
Reads the COMPLETE state from all prior agents and synthesizes
everything into a professional investment brief.

Key differences from other agents:
    - No tool calls — pure LLM synthesis only
    - Highest temperature (0.4) — more natural prose
    - Receives full context from all 3 prior agents
    - Outputs both a Markdown report AND a structured recommendation field
"""

import re
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from core.config import settings, FinancialAnalysisState

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are the Head of Equity Research at a prestigious investment bank.

You have received analysis from three specialist teams and must synthesize
it into a professional, client-ready investment brief.

REPORT STRUCTURE — follow this exactly:

---
## INVESTMENT BRIEF: [COMPANY NAME] ([TICKER])

### EXECUTIVE SUMMARY
[2-3 sentences: what kind of company, key takeaway for an investor]

### MARKET SNAPSHOT
[Current price, daily change, 52-week position, market cap]

### RISK PROFILE
[Volatility, beta, Sharpe, VaR in plain English]

### FUNDAMENTAL ANALYSIS
[Valuation, profitability, growth, balance sheet strength]

### INVESTMENT THESIS
**Bull Case:** [2-3 reasons the stock could outperform]
**Bear Case:** [2-3 risks that could hurt the investment]

### RECOMMENDATION
**[BUY / HOLD / SELL]** | Confidence: [HIGH / MEDIUM / LOW]

*Rationale:* [1-2 sentences explaining the call]
*Suitable for:* [type of investor this stock fits]

*Disclaimer: AI-generated analysis for educational purposes only.
Not financial advice. Always consult a licensed financial advisor.*
---

TONE: professional but accessible — avoid jargon without explanation.
NUMBERS: always cite specific numbers from the data provided.
BALANCE: acknowledge both strengths and weaknesses honestly.
"""


def report_writer_agent_node(state: FinancialAnalysisState) -> dict:
    """
    Synthesizes all prior agents' outputs into the final investment brief.

    No tool loop needed here — single LLM call with full context injected.

    Args:
        state: complete shared state — all agents' outputs are available

    Returns:
        partial state update with 'final_report', 'recommendation',
        and 'analysis_complete' set to True
    """
    ticker       = state.get("ticker", "UNKNOWN").upper()
    company_name = state.get("company_name", ticker)

    logger.info(f"[REPORT WRITER] Synthesizing final report for {company_name}")

    # --- Initialise LLM (no bind_tools — this agent calls no tools) ---
    llm = ChatOpenAI(
        model       = settings.llm_model,
        temperature = settings.writer_temperature,   # 0.4: readable prose
        api_key     = settings.openai_api_key,
    )

    # --- Pull everything from state ---
    market_data  = state.get("market_data", {})
    risk_metrics = state.get("risk_metrics", {})
    fund_data    = state.get("fundamental_data", {})

    # Format the insight lists as readable bullet blocks
    risk_insights_text = "\n".join(
        f"  • {i}" for i in state.get("risk_insights", []) if i.strip()
    ) or "  • No risk insights available"

    fund_insights_text = "\n".join(
        f"  • {i}" for i in state.get("fundamental_insights", []) if i.strip()
    ) or "  • No fundamental insights available"

    # --- Build the comprehensive data brief ---
    # This is the "document" the report writer reads before writing.
    # We format every metric clearly so the LLM doesn't have to guess.
    data_brief = f"""
COMPANY:
  Name     : {company_name}
  Ticker   : {ticker}
  Sector   : {market_data.get('sector', 'N/A')}
  Industry : {market_data.get('industry', 'N/A')}
  About    : {market_data.get('description', 'N/A')[:200]}

MARKET DATA:
  Current price  : ${market_data.get('current_price', 'N/A')}
  Daily change   : ${market_data.get('price_change', 'N/A')} ({market_data.get('price_change_pct', 'N/A')}%)
  Day range      : ${market_data.get('day_low', 'N/A')} – ${market_data.get('day_high', 'N/A')}
  52-week range  : ${market_data.get('week_52_low', 'N/A')} – ${market_data.get('week_52_high', 'N/A')}
  Volume         : {market_data.get('volume', 'N/A'):,}
  Market cap     : ${market_data.get('market_cap', 0):,.0f}

RISK METRICS:
  Risk level     : {risk_metrics.get('risk_level', 'N/A')}
  Ann. volatility: {risk_metrics.get('annualized_volatility_pct', 'N/A')}%
  Beta           : {risk_metrics.get('beta', 'N/A')}
  Sharpe ratio   : {risk_metrics.get('sharpe_ratio', 'N/A')}
  VaR (95% daily): {risk_metrics.get('var_95_daily_pct', 'N/A')}%
  Max drawdown   : {risk_metrics.get('max_drawdown_pct', 'N/A')}%

RISK ANALYST INSIGHTS:
{risk_insights_text}

FUNDAMENTAL METRICS:
  P/E (TTM)      : {fund_data.get('pe_ratio', 'N/A')}x
  Forward P/E    : {fund_data.get('forward_pe', 'N/A')}x
  PEG ratio      : {fund_data.get('peg_ratio', 'N/A')}
  EPS (TTM)      : ${fund_data.get('eps_ttm', 'N/A')}
  Revenue growth : {fund_data.get('revenue_growth_pct', 'N/A')}%
  Gross margin   : {fund_data.get('gross_margin_pct', 'N/A')}%
  Net margin     : {fund_data.get('net_margin_pct', 'N/A')}%
  ROE            : {fund_data.get('roe_pct', 'N/A')}%
  Debt/Equity    : {fund_data.get('debt_to_equity', 'N/A')}x
  Current ratio  : {fund_data.get('current_ratio', 'N/A')}x
  Free cash flow : ${fund_data.get('free_cash_flow_usd', 0):,.0f}
  Dividend yield : {fund_data.get('dividend_yield_pct', 'N/A')}%

FUNDAMENTAL ANALYST INSIGHTS:
{fund_insights_text}

ORIGINAL REQUEST: {state.get('analysis_request', 'General investment analysis')}
"""

    # --- Single LLM call — no tool loop needed ---
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=f"Write the investment brief using this data:\n{data_brief}"
        ),
    ]

    logger.info("[REPORT WRITER] Calling LLM for synthesis...")
    response     = llm.invoke(messages)
    final_report = response.content

    # --- Extract structured recommendation from the report text ---
    # The UI needs BUY/HOLD/SELL as a clean field, not buried in Markdown.
    # We use regex to find the pattern **BUY** or **HOLD** or **SELL**
    recommendation = "HOLD"
    confidence     = "MEDIUM"

    rec_match = re.search(r'\*{0,2}(BUY|HOLD|SELL)\*{0,2}', final_report, re.IGNORECASE)
    if rec_match:
        recommendation = rec_match.group(1).upper()

    conf_match = re.search(r'Confidence:\s*\*{0,2}(HIGH|MEDIUM|LOW)\*{0,2}', final_report, re.IGNORECASE)
    if conf_match:
        confidence = conf_match.group(1).upper()

    full_recommendation = f"{recommendation} | Confidence: {confidence}"

    logger.info(f"[REPORT WRITER] Done. Recommendation: {full_recommendation}")

    return {
        "final_report"      : final_report,
        "recommendation"    : full_recommendation,
        "analysis_complete" : True,            # signals supervisor to route to END
        "agent_steps"       : [
            f"REPORT_WRITER: {len(final_report)} chars | "
            f"recommendation={full_recommendation}"
        ],
    }