"""
tools/financial_tools.py — Financial Data & Calculation Tools

Agents are the decision-makers. Tools are their hands.
The @tool decorator makes each function callable by an LLM agent.
"""

import math
import logging
from datetime import datetime
from typing import Any

from langchain_core.tools import tool
import yfinance as yf

logger = logging.getLogger(__name__)


# =============================================================================
# TOOL 1 — Fetch Stock Price
# =============================================================================

@tool
def fetch_stock_price(ticker: str) -> dict[str, Any]:
    """
    Fetch current stock price and recent OHLCV data from Yahoo Finance.

    Returns current price, daily change, 52-week range,
    volume, market cap, and 30 days of closing prices.
    """
    logger.info(f"[TOOL] fetch_stock_price → {ticker}")
    try:
        stock = yf.Ticker(ticker.upper()) # AAPL,ORCL,TSLA
        info  = stock.fast_info

        # Download 30 days of daily bars for historical analysis
        hist = stock.history(period="1mo", interval="1d")
        closing_prices = hist["Close"].tolist() if not hist.empty else []

        result = {
            "ticker"            : ticker.upper(),
            "timestamp"         : datetime.now().isoformat(),
            "current_price"     : round(float(info.last_price or 0), 2),
            "previous_close"    : round(float(info.previous_close or 0), 2),
            "price_change"      : round(float((info.last_price or 0) - (info.previous_close or 0)), 2),
            "price_change_pct"  : round(
                                    float(((info.last_price or 0) - (info.previous_close or 0))
                                    / (info.previous_close or 1) * 100), 2),
            "day_high"          : round(float(info.day_high or 0), 2),
            "day_low"           : round(float(info.day_low or 0), 2),
            "week_52_high"      : round(float(info.year_high or 0), 2),
            "week_52_low"       : round(float(info.year_low or 0), 2),
            "volume"            : int(info.last_volume or 0),
            "avg_volume"        : int(info.three_month_average_volume or 0),
            "market_cap"        : float(info.market_cap or 0),
            "historical_prices" : [round(p, 2) for p in closing_prices],
        }

        logger.info(f"[TOOL] fetch_stock_price ✅ {ticker} @ ${result['current_price']}")
        return result

    except Exception as e:
        logger.error(f"[TOOL] fetch_stock_price FAILED for {ticker}: {e}")
        return {"ticker": ticker, "error": str(e), "timestamp": datetime.now().isoformat()}


# =============================================================================
# TOOL 2 — Fetch Company Info
# =============================================================================

@tool
def fetch_company_info(ticker: str) -> dict[str, Any]:
    """
    Fetch company metadata: name, sector, industry, description, employees.

    Knowing the sector matters because valuation benchmarks differ
    by industry (e.g. tech P/E of 30 is normal, utilities P/E of 30 is not).
    """
    logger.info(f"[TOOL] fetch_company_info → {ticker}")
    try:
        stock = yf.Ticker(ticker.upper())
        info  = stock.info   # heavier call than fast_info — has full metadata

        result = {
            "ticker"       : ticker.upper(),
            "company_name" : info.get("longName", "Unknown"),
            "sector"       : info.get("sector", "Unknown"),
            "industry"     : info.get("industry", "Unknown"),
            "country"      : info.get("country", "Unknown"),
            "employees"    : info.get("fullTimeEmployees", 0),
            "website"      : info.get("website", ""),
            "description"  : info.get("longBusinessSummary", "")[:500],
            "exchange"     : info.get("exchange", "Unknown"),
            "currency"     : info.get("currency", "USD"),
        }

        logger.info(f"[TOOL] fetch_company_info ✅ {result['company_name']}")
        return result

    except Exception as e:
        logger.error(f"[TOOL] fetch_company_info FAILED for {ticker}: {e}")
        return {"ticker": ticker, "error": str(e)}


# =============================================================================
# TOOL 3 — Calculate Risk Metrics
# =============================================================================

@tool
def calculate_risk_metrics(ticker: str) -> dict[str, Any]:
    """
    Calculate key risk metrics using 1 year of daily price history.

    METRICS EXPLAINED:

    VOLATILITY (annualized std dev of returns)
        How much does the price swing?
        Formula: σ_daily × √252   (252 = trading days in a year)
        > 40% = HIGH risk,  20–40% = MEDIUM,  < 20% = LOW

    BETA
        How does the stock move vs the S&P 500?
        Beta = Cov(stock, market) / Var(market)
        1.0 = moves with market,  > 1.0 = more volatile,  < 1.0 = defensive

    SHARPE RATIO
        Return per unit of risk. Higher is better.
        Formula: (avg return − risk free rate) / std dev  ×  √252
        > 2.0 = excellent,  1–2 = acceptable,  < 1 = poor

    VALUE AT RISK (VaR) at 95% confidence
        "On 95% of days, we will NOT lose more than X%"
        Formula: mean return − 1.645 × std dev
        1.645 is the z-score for the 95th percentile
    """
    logger.info(f"[TOOL] calculate_risk_metrics → {ticker}")
    try:
        stock = yf.Ticker(ticker.upper())

        # Need at least 1 year of data for meaningful statistics
        hist = stock.history(period="1y", interval="1d")

        if hist.empty or len(hist) < 10:
            return {"ticker": ticker, "error": "Insufficient historical data"}

        # Daily return = (today − yesterday) / yesterday
        # pct_change() does this for every row, dropna() removes the first NaN
        daily_returns = hist["Close"].pct_change().dropna()
        returns_list  = daily_returns.tolist()
        n             = len(returns_list)

        # Mean daily return
        mean_return = sum(returns_list) / n

        # Variance = average squared deviation from mean
        variance  = sum((r - mean_return) ** 2 for r in returns_list) / (n - 1)
        daily_std = math.sqrt(variance)

        # Annualised volatility: multiply by √252
        annualized_volatility = daily_std * math.sqrt(252)

        # --- Beta ---
        beta = None
        try:
            spy      = yf.Ticker("^GSPC")           # S&P 500 as benchmark
            spy_hist = spy.history(period="1y", interval="1d")

            if not spy_hist.empty:
                import pandas as pd
                spy_returns = spy_hist["Close"].pct_change().dropna()

                # Align both series by date (inner join drops mismatched dates)
                aligned = pd.concat([daily_returns, spy_returns], axis=1, join="inner")
                aligned.columns = ["stock", "market"]

                if len(aligned) > 10:
                    s  = aligned["stock"].tolist()
                    m  = aligned["market"].tolist()
                    n2 = len(s)
                    ms = sum(s) / n2
                    mm = sum(m) / n2

                    # Covariance: how stock and market move together
                    cov = sum((s[i]-ms)*(m[i]-mm) for i in range(n2)) / (n2-1)
                    # Market variance
                    var_m = sum((m[i]-mm)**2 for i in range(n2)) / (n2-1)

                    beta = round(cov / var_m, 3) if var_m != 0 else None
        except Exception:
            pass   # Beta fails gracefully — we still return other metrics

        # --- Sharpe Ratio ---
        # Risk-free rate: ~5% annual → 5/252 daily
        rf_daily      = 0.05 / 252
        excess        = [r - rf_daily for r in returns_list]
        mean_excess   = sum(excess) / len(excess)
        sharpe_ratio  = (mean_excess / daily_std) * math.sqrt(252) if daily_std > 0 else 0

        # --- VaR at 95% confidence ---
        # 1.645 = z-score where 5% of normal distribution sits to the left
        var_95 = mean_return - (1.645 * daily_std)

        # --- Max Drawdown ---
        # Largest peak-to-trough loss over the period
        prices       = hist["Close"].tolist()
        peak         = prices[0]
        max_drawdown = 0.0
        for price in prices:
            if price > peak:
                peak = price
            drawdown = (peak - price) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        result = {
            "ticker"                    : ticker.upper(),
            "annualized_volatility_pct" : round(annualized_volatility * 100, 2),
            "daily_volatility_pct"      : round(daily_std * 100, 4),
            "beta"                      : beta,
            "sharpe_ratio"              : round(sharpe_ratio, 3),
            "var_95_daily_pct"          : round(var_95 * 100, 3),
            "max_drawdown_pct"          : round(max_drawdown * 100, 2),
            "avg_daily_return_pct"      : round(mean_return * 100, 4),
            "trading_days_analyzed"     : n,
            "risk_level"                : (
                "HIGH"   if annualized_volatility > 0.40 else
                "MEDIUM" if annualized_volatility > 0.20 else
                "LOW"
            ),
        }

        logger.info(f"[TOOL] calculate_risk_metrics ✅ vol={result['annualized_volatility_pct']}%")
        return result

    except Exception as e:
        logger.error(f"[TOOL] calculate_risk_metrics FAILED for {ticker}: {e}")
        return {"ticker": ticker, "error": str(e)}


# =============================================================================
# TOOL 4 — Fetch Fundamental Data
# =============================================================================

@tool
def fetch_fundamental_data(ticker: str) -> dict[str, Any]:
    """
    Fetch fundamental financial metrics from Yahoo Finance.

    KEY CONCEPTS:

    P/E RATIO  = Price / EPS  — "how much are you paying for $1 of earnings?"
    PEG RATIO  = P/E / Growth — PEG < 1 means growth justifies the price
    NET MARGIN = Net Income / Revenue — bottom-line profitability
    ROE        = Net Income / Equity — how well is management using your money?
    DEBT/EQUITY = Total Debt / Equity — higher = more financial risk
    FREE CASH FLOW = real cash after capex — the truest measure of profitability
    """
    logger.info(f"[TOOL] fetch_fundamental_data → {ticker}")
    try:
        stock = yf.Ticker(ticker.upper())
        info  = stock.info

        # Safe getter — returns 0.0 if key is missing or None
        def g(key: str, default: float = 0.0) -> float:
            val = info.get(key, default)
            return float(val) if val is not None else default

        result = {
            "ticker"              : ticker.upper(),
            # Valuation
            "pe_ratio"            : round(g("trailingPE"), 2),
            "forward_pe"          : round(g("forwardPE"), 2),
            "pb_ratio"            : round(g("priceToBook"), 2),
            "ps_ratio"            : round(g("priceToSalesTrailingTwelveMonths"), 2),
            "peg_ratio"           : round(g("pegRatio"), 2),
            "eps_ttm"             : round(g("trailingEps"), 2),
            "eps_forward"         : round(g("forwardEps"), 2),
            # Growth
            "revenue_growth_pct"  : round(g("revenueGrowth") * 100, 2),
            "earnings_growth_pct" : round(g("earningsGrowth") * 100, 2),
            "revenue_ttm_usd"     : round(g("totalRevenue"), 0),
            # Profitability
            "gross_margin_pct"    : round(g("grossMargins") * 100, 2),
            "operating_margin_pct": round(g("operatingMargins") * 100, 2),
            "net_margin_pct"      : round(g("profitMargins") * 100, 2),
            "roe_pct"             : round(g("returnOnEquity") * 100, 2),
            "roa_pct"             : round(g("returnOnAssets") * 100, 2),
            "ebitda_usd"          : round(g("ebitda"), 0),
            # Financial Health
            "debt_to_equity"      : round(g("debtToEquity"), 2),
            "current_ratio"       : round(g("currentRatio"), 2),
            "quick_ratio"         : round(g("quickRatio"), 2),
            "free_cash_flow_usd"  : round(g("freeCashflow"), 0),
            # Dividend
            "dividend_yield_pct"  : round(g("dividendYield") * 100, 2),
            "payout_ratio_pct"    : round(g("payoutRatio") * 100, 2),
        }

        logger.info(f"[TOOL] fetch_fundamental_data ✅ PE={result['pe_ratio']}")
        return result

    except Exception as e:
        logger.error(f"[TOOL] fetch_fundamental_data FAILED for {ticker}: {e}")
        return {"ticker": ticker, "error": str(e)}


# =============================================================================
# Tool registries — imported by agents to register their available tools
# =============================================================================
# Each agent only gets the tools relevant to its job.
# This is good practice: don't give an agent tools it shouldn't use.

ALL_MARKET_TOOLS      = [fetch_stock_price, fetch_company_info]
ALL_RISK_TOOLS        = [calculate_risk_metrics]
ALL_FUNDAMENTAL_TOOLS = [fetch_fundamental_data]
ALL_TOOLS             = [fetch_stock_price, fetch_company_info,
                         calculate_risk_metrics, fetch_fundamental_data]

