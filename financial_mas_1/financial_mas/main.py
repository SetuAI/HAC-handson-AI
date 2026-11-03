"""
main.py — Command Line Entry Point

Runs the full multi-agent pipeline from the terminal.

Usage:
    python main.py
    python main.py --ticker MSFT
    python main.py --ticker TSLA --request "Focus on EV growth"
"""

import os
import sys
import logging
import argparse
from datetime import datetime

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def get_args():
    parser = argparse.ArgumentParser(
        prog       = "financial-mas",
        description= "Financial Multi-Agent System — LangGraph + OpenAI",
    )
    parser.add_argument(
        "--ticker",
        type   = str,
        default= "AAPL",
        help   = "Stock ticker to analyse (default: AAPL)",
    )
    parser.add_argument(
        "--request",
        type   = str,
        default= "Provide a comprehensive investment analysis with buy/hold/sell recommendation",
        help   = "Custom analysis instruction",
    )
    return parser.parse_args()


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║        FINANCIAL MULTI-AGENT SYSTEM                         ║
║        LangGraph × OpenAI × yfinance                        ║
╚══════════════════════════════════════════════════════════════╝
    """)


def section(title: str):
    print(f"\n{'─' * 65}")
    print(f"  {title}")
    print(f"{'─' * 65}")


def print_metrics(state: dict):
    """Print a clean summary table of key metrics."""
    section("📊 KEY METRICS")

    md = state.get("market_data", {})
    rm = state.get("risk_metrics", {})
    fd = state.get("fundamental_data", {})

    print("\n  MARKET DATA:")
    print(f"    Price         : ${md.get('current_price', 'N/A')}")
    chg = md.get('price_change_pct', 0)
    print(f"    Daily change  : {chg:+.2f}%" if isinstance(chg, float) else "    Daily change  : N/A")
    cap = md.get('market_cap', 0)
    print(f"    Market cap    : ${cap/1e9:.2f}B" if cap else "    Market cap    : N/A")
    print(f"    52-week range : ${md.get('week_52_low','N/A')} – ${md.get('week_52_high','N/A')}")

    print("\n  RISK METRICS:")
    print(f"    Risk level    : {rm.get('risk_level', 'N/A')}")
    print(f"    Volatility    : {rm.get('annualized_volatility_pct', 'N/A')}% (annualised)")
    print(f"    Beta          : {rm.get('beta', 'N/A')}")
    print(f"    Sharpe ratio  : {rm.get('sharpe_ratio', 'N/A')}")
    print(f"    VaR (95% day) : {rm.get('var_95_daily_pct', 'N/A')}%")
    print(f"    Max drawdown  : {rm.get('max_drawdown_pct', 'N/A')}%")

    print("\n  FUNDAMENTALS:")
    print(f"    P/E ratio     : {fd.get('pe_ratio', 'N/A')}x")
    print(f"    EPS (TTM)     : ${fd.get('eps_ttm', 'N/A')}")
    print(f"    Net margin    : {fd.get('net_margin_pct', 'N/A')}%")
    print(f"    Revenue growth: {fd.get('revenue_growth_pct', 'N/A')}%")
    print(f"    Debt/Equity   : {fd.get('debt_to_equity', 'N/A')}x")
    print(f"    ROE           : {fd.get('roe_pct', 'N/A')}%")


def print_insights(state: dict):
    """Print agent insights."""
    risk_insights = state.get("risk_insights", [])
    if risk_insights:
        section("⚠️  RISK ANALYST INSIGHTS")
        for insight in risk_insights:
            if insight.strip():
                print(f"  • {insight}")

    fund_insights = state.get("fundamental_insights", [])
    if fund_insights:
        section("📈 FUNDAMENTAL ANALYST INSIGHTS")
        for insight in fund_insights:
            if insight.strip():
                print(f"  • {insight}")


def print_agent_steps(state: dict):
    """Print the agent execution timeline."""
    section("🔍 AGENT EXECUTION TIMELINE")

    icons = {
        "SUPERVISOR"  : "🔵",
        "MARKET_DATA" : "📊",
        "RISK"        : "⚠️ ",
        "FUNDAMENTAL" : "📈",
        "REPORT"      : "📝",
    }

    for i, step in enumerate(state.get("agent_steps", []), 1):
        icon = "  "
        for key, emoji in icons.items():
            if key in step.upper():
                icon = emoji
                break
        print(f"  {i:2d}. {icon} {step}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    args   = get_args()
    ticker = args.ticker.upper().strip()

    banner()

    # Validate API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY is not set.")
        print("   Run: export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    print(f"  Ticker    : {ticker}")
    print(f"  Request   : {args.request[:80]}")
    print(f"  Started   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n  Running pipeline — this takes ~30-60 seconds...\n")

    # Import here so --help is instant
    from graph.pipeline import run_financial_analysis

    start = datetime.now()

    try:
        final_state = run_financial_analysis(
            ticker           = ticker,
            analysis_request = args.request,
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        logger.exception("Traceback:")
        sys.exit(1)

    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n  ✅ Done in {elapsed:.1f} seconds")

    # Print results
    print_agent_steps(final_state)
    print_metrics(final_state)
    print_insights(final_state)

    # Recommendation
    rec = final_state.get("recommendation", "HOLD")
    symbols = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}
    emoji   = symbols.get(rec.split("|")[0].strip(), "⚪")
    section(f"{emoji} RECOMMENDATION: {rec}")

    # Full report
    report = final_state.get("final_report", "")
    if report:
        section("📝 FULL INVESTMENT BRIEF")
        print(report)

    print(f"\n{'═' * 65}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"{'═' * 65}\n")


if __name__ == "__main__":
    main()