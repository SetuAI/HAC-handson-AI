"""
ui/dashboard.py — Streamlit Interactive Dashboard

Provides a visual interface for the Financial Multi-Agent System.
Talks directly to graph/pipeline.py — does not go through FastAPI.

Run with:
    streamlit run ui/dashboard.py
"""

import os
import time
import logging
import streamlit as st
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


logger = logging.getLogger(__name__)

# =============================================================================
# PAGE CONFIG — must be the very first Streamlit command
# =============================================================================

st.set_page_config(
    page_title        = "Financial MAS — LangGraph",
    page_icon         = "📈",
    layout            = "wide",
    initial_sidebar_state = "expanded",
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .rec-buy  { background:#d1e7dd; border-left:5px solid #198754;
                padding:1rem; border-radius:0 8px 8px 0; }
    .rec-hold { background:#fff3cd; border-left:5px solid #ffc107;
                padding:1rem; border-radius:0 8px 8px 0; }
    .rec-sell { background:#f8d7da; border-left:5px solid #dc3545;
                padding:1rem; border-radius:0 8px 8px 0; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPERS
# =============================================================================

def format_number(num: float) -> str:
    """Format large numbers: 3140000000000 → $3.14T"""
    if not num or num == 0:
        return "N/A"
    if abs(num) >= 1e12:
        return f"${num/1e12:.2f}T"
    if abs(num) >= 1e9:
        return f"${num/1e9:.2f}B"
    if abs(num) >= 1e6:
        return f"${num/1e6:.2f}M"
    return f"${num:,.2f}"


def rec_css_class(rec: str) -> str:
    """Return CSS class based on recommendation."""
    if "BUY"  in rec.upper(): return "rec-buy"
    if "SELL" in rec.upper(): return "rec-sell"
    return "rec-hold"


def risk_emoji(risk: str) -> str:
    return {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(
        str(risk).upper(), "⚪"
    )


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # API key input
    api_key = st.text_input(
        "OpenAI API Key",
        value       = os.getenv("OPENAI_API_KEY", ""),
        type        = "password",
        placeholder = "sk-...",
        help        = "Or set OPENAI_API_KEY in your .env file",
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.divider()
    st.markdown("## 🎯 Quick Tickers")

    # Quick select buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("AAPL"): st.session_state["ticker"] = "AAPL"
        if st.button("TSLA"): st.session_state["ticker"] = "TSLA"
    with col2:
        if st.button("MSFT"): st.session_state["ticker"] = "MSFT"
        if st.button("GOOG"): st.session_state["ticker"] = "GOOG"
    with col3:
        if st.button("NVDA"): st.session_state["ticker"] = "NVDA"
        if st.button("AMZN"): st.session_state["ticker"] = "AMZN"

    st.divider()
    st.markdown("## 🤖 Agent Pipeline")
    st.info(
        "🔵 Supervisor\n\n"
        "📊 Market Data Agent\n\n"
        "⚠️ Risk Analyst Agent\n\n"
        "📈 Fundamental Analyst\n\n"
        "📝 Report Writer Agent"
    )


# =============================================================================
# HEADER
# =============================================================================

st.markdown("""
<div class="main-header">
    <h1>📈 Financial Multi-Agent System</h1>
    <p style="margin:0;opacity:0.8">LangGraph × OpenAI × yfinance — Educational Demo</p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# INPUT ROW
# =============================================================================

col_ticker, col_request = st.columns([1, 2])

with col_ticker:
    ticker = st.text_input(
        "📌 Stock Ticker",
        value       = st.session_state.get("ticker", "AAPL"),
        placeholder = "e.g. AAPL, MSFT, TSLA",
    ).upper().strip()

with col_request:
    analysis_request = st.text_area(
        "💬 Analysis Request",
        value  = "Provide a comprehensive investment analysis with buy/hold/sell recommendation",
        height = 80,
    )

# Run button
run_col, info_col = st.columns([1, 3])

with run_col:
    run_button = st.button(
        "🚀 Run Analysis",
        type             = "primary",
        use_container_width = True,
        disabled         = not os.getenv("OPENAI_API_KEY"),
    )

with info_col:
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("⚠️ Enter your OpenAI API key in the sidebar to run analysis")
    else:
        st.info(f"Ready to analyse **{ticker}** with 4 specialist AI agents. ~30-60 seconds.")


# =============================================================================
# RUN THE PIPELINE
# =============================================================================

if run_button and ticker and os.getenv("OPENAI_API_KEY"):

    st.markdown("---")
    st.markdown("### 🔄 Agent Pipeline — Live Progress")

    # Progress bar
    progress_bar = st.progress(0, text="Initialising pipeline...")

    # One status placeholder per agent
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: s_supervisor   = st.empty(); s_supervisor.markdown("🔵 **Supervisor**\n\n⏳ Waiting")
    with c2: s_market       = st.empty(); s_market.markdown("📊 **Market Data**\n\n⏳ Waiting")
    with c3: s_risk         = st.empty(); s_risk.markdown("⚠️ **Risk Analyst**\n\n⏳ Waiting")
    with c4: s_fundamental  = st.empty(); s_fundamental.markdown("📈 **Fundamental**\n\n⏳ Waiting")
    with c5: s_report       = st.empty(); s_report.markdown("📝 **Report Writer**\n\n⏳ Waiting")

    # Live log
    with st.expander("📋 Live Agent Logs", expanded=True):
        log_box  = st.empty()
        log_lines = []

    try:
        # Import here so startup is fast
        from graph.pipeline import stream_financial_analysis

        start      = time.time()
        final_state = {}

        # Stream the pipeline — one chunk per agent completing
        for chunk in stream_financial_analysis(ticker, analysis_request):
            for node_name, update in chunk.items():

                # Update progress bar
                pct_map = {
                    "supervisor"                : (15,  "Supervisor routing..."),
                    "market_data_agent"         : (35,  "Market Data Agent fetching..."),
                    "risk_analyst_agent"        : (60,  "Risk Analyst calculating..."),
                    "fundamental_analyst_agent" : (80,  "Fundamental Analyst evaluating..."),
                    "report_writer_agent"       : (95,  "Report Writer synthesizing..."),
                }
                if node_name in pct_map:
                    pct, msg = pct_map[node_name]
                    progress_bar.progress(pct, text=f"🔄 {msg}")

                # Update agent status badges
                if node_name == "supervisor":
                    s_supervisor.markdown("🔵 **Supervisor**\n\n✅ Done")
                elif node_name == "market_data_agent":
                    s_market.markdown("📊 **Market Data**\n\n✅ Done")
                elif node_name == "risk_analyst_agent":
                    s_risk.markdown("⚠️ **Risk Analyst**\n\n✅ Done")
                elif node_name == "fundamental_analyst_agent":
                    s_fundamental.markdown("📈 **Fundamental**\n\n✅ Done")
                elif node_name == "report_writer_agent":
                    s_report.markdown("📝 **Report Writer**\n\n✅ Done")

                # Append to live log
                for step in update.get("agent_steps", []):
                    log_lines.append(f"✅ {step}")
                if log_lines:
                    log_box.code("\n".join(log_lines[-20:]))

                # Merge state updates
                for k, v in update.items():
                    if isinstance(v, list) and isinstance(final_state.get(k), list):
                        final_state[k] = final_state.get(k, []) + v
                    elif v:
                        final_state[k] = v

        elapsed = time.time() - start
        progress_bar.progress(100, text=f"✅ Complete in {elapsed:.1f}s")
        st.session_state["results"] = final_state

    except Exception as e:
        st.error(f"❌ Pipeline failed: {e}")
        st.exception(e)


# =============================================================================
# RESULTS
# =============================================================================

if st.session_state.get("results"):
    state = st.session_state["results"]

    st.markdown("---")
    st.markdown(f"## 📊 Results: {state.get('company_name', ticker)} ({ticker})")

    # --- Top row: recommendation + key metrics ---
    rec_col, stats_col = st.columns([1, 2])

    with rec_col:
        rec       = state.get("recommendation", "HOLD | Confidence: MEDIUM")
        rec_class = rec_css_class(rec)
        action    = rec.split("|")[0].strip()
        conf      = rec.split("|")[1].strip() if "|" in rec else ""
        st.markdown(
            f'<div class="{rec_class}"><h2 style="margin:0">{action}</h2>'
            f'<p style="margin:0.3rem 0 0 0">{conf}</p></div>',
            unsafe_allow_html=True
        )

    with stats_col:
        md   = state.get("market_data", {})
        rm   = state.get("risk_metrics", {})
        fd   = state.get("fundamental_data", {})
        m1, m2, m3, m4 = st.columns(4)

        price   = md.get("current_price", 0)
        chg_pct = md.get("price_change_pct", 0)
        with m1: st.metric("Price",      f"${price:.2f}" if price else "N/A",
                            f"{chg_pct:+.2f}%" if isinstance(chg_pct, float) else None)
        with m2: st.metric("Market Cap", format_number(md.get("market_cap", 0)))
        with m3:
            rl = rm.get("risk_level", "N/A")
            st.metric("Risk",  f"{risk_emoji(rl)} {rl}")
        with m4:
            pe = fd.get("pe_ratio", 0)
            st.metric("P/E",   f"{pe:.1f}x" if pe else "N/A")

    st.markdown("---")

    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Investment Report",
        "⚠️ Risk Metrics",
        "📈 Fundamentals",
        "🔍 Agent Steps",
    ])

    # Tab 1 — Full report
    with tab1:
        st.markdown(state.get("final_report", "No report generated"))

    # Tab 2 — Risk metrics
    with tab2:
        risk = state.get("risk_metrics", {})
        if risk:
            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown("### Volatility")
                st.metric("Annualised Vol",  f"{risk.get('annualized_volatility_pct',0):.1f}%")
                st.metric("Max Drawdown",    f"-{risk.get('max_drawdown_pct',0):.1f}%")
            with r2:
                st.markdown("### Risk-Adjusted")
                st.metric("Sharpe Ratio",    f"{risk.get('sharpe_ratio',0):.2f}")
                st.metric("Beta",            f"{risk.get('beta','N/A')}")
            with r3:
                st.markdown("### Value at Risk")
                st.metric("VaR 95% (daily)", f"{risk.get('var_95_daily_pct',0):.3f}%")
                st.metric("Avg Daily Return",f"{risk.get('avg_daily_return_pct',0):.3f}%")

            st.markdown("### 💡 Risk Insights")
            for insight in state.get("risk_insights", []):
                if insight.strip():
                    st.markdown(f"• {insight}")

    # Tab 3 — Fundamentals
    with tab3:
        fund = state.get("fundamental_data", {})
        if fund:
            f1, f2, f3 = st.columns(3)
            with f1:
                st.markdown("### Valuation")
                st.metric("P/E (TTM)",    f"{fund.get('pe_ratio',0):.1f}x")
                st.metric("Forward P/E",  f"{fund.get('forward_pe',0):.1f}x")
                st.metric("PEG Ratio",    f"{fund.get('peg_ratio',0):.2f}")
                st.metric("EPS (TTM)",    f"${fund.get('eps_ttm',0):.2f}")
            with f2:
                st.markdown("### Profitability")
                st.metric("Gross Margin",  f"{fund.get('gross_margin_pct',0):.1f}%")
                st.metric("Net Margin",    f"{fund.get('net_margin_pct',0):.1f}%")
                st.metric("ROE",           f"{fund.get('roe_pct',0):.1f}%")
            with f3:
                st.markdown("### Financial Health")
                st.metric("Debt/Equity",   f"{fund.get('debt_to_equity',0):.2f}x")
                st.metric("Current Ratio", f"{fund.get('current_ratio',0):.2f}x")
                st.metric("Free Cash Flow",format_number(fund.get("free_cash_flow_usd",0)))

            st.markdown("### 💡 Fundamental Insights")
            for insight in state.get("fundamental_insights", []):
                if insight.strip():
                    st.markdown(f"• {insight}")

    # Tab 4 — Agent steps
    with tab4:
        st.markdown("### Agent Execution Timeline")
        st.info("Chronological log of every decision made by each agent.")

        icon_map = {
            "SUPERVISOR": "🔵", "MARKET_DATA": "📊",
            "RISK": "⚠️", "FUNDAMENTAL": "📈", "REPORT": "📝",
        }
        for i, step in enumerate(state.get("agent_steps", []), 1):
            icon = "▪️"
            for key, emoji in icon_map.items():
                if key in step.upper():
                    icon = emoji
                    break
            st.markdown(f"**{i}.** {icon} {step}")

        with st.expander("🔧 Raw State (debug)"):
            st.json({
                k: v for k, v in state.items()
                if k not in ["messages", "historical_prices", "final_report"]
            })


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#6c757d;font-size:0.85rem'>"
    "📚 Financial MAS — Educational Demo | "
    "Built with LangGraph + OpenAI + Streamlit<br>"
    "⚠️ Not financial advice. For educational purposes only."
    "</div>",
    unsafe_allow_html=True,
)