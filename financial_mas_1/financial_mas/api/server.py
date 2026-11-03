"""
api/server.py — FastAPI REST Gateway

Exposes the LangGraph pipeline as a REST API.

uvicorn api.server:app --reload --port 8000

ENDPOINTS:
    POST /analyze          → full analysis, synchronous (~30-60s)
    GET  /analyze/stream   → streaming analysis via SSE
    GET  /metrics/{ticker} → raw metrics, no LLM, fast (2-5s)
    GET  /health           → health check
    GET  /docs             → auto-generated Swagger UI (free from FastAPI)

WHY FASTAPI:
    - Pydantic models give automatic request validation
    - Auto-generates /docs Swagger UI — great for demos and teaching
    - Async support handles concurrent requests cleanly
    - Type hints everywhere — readable, self-documenting code
"""

import json
import logging
import time
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator

from core.config import settings
from graph.pipeline import run_financial_analysis, stream_financial_analysis

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# APP INSTANCE
# =============================================================================

app = FastAPI(
    title       = "Financial Multi-Agent Analysis API",
    description = (
        "LangGraph-powered multi-agent system for financial analysis. "
        "Runs Market Data → Risk Analyst → Fundamental Analyst → Report Writer."
    ),
    version  = settings.app_version,
    docs_url = "/docs",    # Swagger UI  → http://localhost:8000/docs
    redoc_url= "/redoc",   # ReDoc UI    → http://localhost:8000/redoc
)

# CORS — allows the Streamlit UI (port 8501) to call this API (port 8000)
# Without this, browsers block cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # restrict to specific domains in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# =============================================================================
# PYDANTIC REQUEST / RESPONSE MODELS
# =============================================================================
# These serve two purposes:
#   1. Validation  — FastAPI rejects malformed requests automatically
#   2. Docs        — FastAPI reads these to generate the Swagger schema

class AnalysisRequest(BaseModel):
    """Request body for POST /analyze"""

    ticker: str = Field(
        ...,                 # required — no default
        min_length = 1,
        max_length = 10,
        description= "Stock ticker symbol e.g. AAPL, MSFT, TSLA",
        example    = "AAPL",
    )
    analysis_request: Optional[str] = Field(
        default    = "Provide a comprehensive investment analysis with buy/hold/sell recommendation",
        description= "Custom analysis instruction or focus area",
        example    = "Focus on growth potential for a 5-year horizon",
    )

    @validator("ticker")
    def uppercase_ticker(cls, v):
        """Auto-convert ticker to uppercase — aapl, AAPL, Aapl all work"""
        return v.upper().strip()


class AnalysisResponse(BaseModel):
    """Response body for POST /analyze"""

    # Metadata
    ticker                 : str
    company_name           : str
    analysis_timestamp     : str
    execution_time_seconds : float
    status                 : str            # "success" or "error"

    # Core outputs
    recommendation : str    # "BUY | Confidence: HIGH"
    final_report   : str    # full Markdown investment brief

    # Key metrics (flattened from nested dicts for convenience)
    current_price  : Optional[float]
    market_cap     : Optional[float]
    risk_level     : Optional[str]
    pe_ratio       : Optional[float]

    # Audit trail
    agent_steps : list[str]
    errors      : list[str]


# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.get("/health", tags=["System"], summary="Health check")
async def health_check():
    """
    Returns 200 OK if the server is running.
    Used by load balancers, Kubernetes probes, and monitoring tools.
    """
    return {
        "status"    : "healthy",
        "timestamp" : datetime.now().isoformat(),
        "version"   : settings.app_version,
        "model"     : settings.llm_model,
    }


# =============================================================================
# MAIN ANALYSIS ENDPOINT
# =============================================================================

@app.post(
    "/analyze",
    response_model = AnalysisResponse,
    tags           = ["Analysis"],
    summary        = "Run full multi-agent financial analysis",
)
async def analyze_stock(request: AnalysisRequest):
    """
    Triggers the full LangGraph pipeline and waits for completion.

    Flow:
        1. FastAPI validates the request via AnalysisRequest
        2. Pipeline runs: supervisor → 4 agents → END
        3. Final state is packaged into AnalysisResponse
        4. FastAPI validates the response via AnalysisResponse
        5. JSON is returned to the caller

    Typical execution time: 30-60 seconds.
    For real-time progress use GET /analyze/stream instead.
    """
    logger.info(f"[API] POST /analyze → ticker={request.ticker}")
    start_time = time.time()

    try:
        final_state    = run_financial_analysis(
            ticker           = request.ticker,
            analysis_request = request.analysis_request,
        )
        execution_time = time.time() - start_time
        logger.info(f"[API] Analysis done in {execution_time:.1f}s")

        market_data  = final_state.get("market_data", {})
        risk_metrics = final_state.get("risk_metrics", {})
        fund_data    = final_state.get("fundamental_data", {})

        return AnalysisResponse(
            ticker                 = request.ticker,
            company_name           = final_state.get("company_name", request.ticker),
            analysis_timestamp     = datetime.now().isoformat(),
            execution_time_seconds = round(execution_time, 2),
            status                 = "success",
            recommendation         = final_state.get("recommendation", "HOLD | Confidence: LOW"),
            final_report           = final_state.get("final_report", "Report generation failed"),
            current_price          = market_data.get("current_price"),
            market_cap             = market_data.get("market_cap"),
            risk_level             = risk_metrics.get("risk_level"),
            pe_ratio               = fund_data.get("pe_ratio"),
            agent_steps            = final_state.get("agent_steps", []),
            errors                 = final_state.get("errors", []),
        )

    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"[API] Analysis failed for {request.ticker}: {e}", exc_info=True)
        raise HTTPException(
            status_code = 500,
            detail      = {
                "error"   : str(e),
                "ticker"  : request.ticker,
                "duration": round(execution_time, 2),
            }
        )


# =============================================================================
# STREAMING ENDPOINT
# =============================================================================

@app.get(
    "/analyze/stream",
    tags    = ["Analysis"],
    summary = "Stream analysis results in real-time (SSE)",
)
async def stream_analysis(
    ticker           : str,
    analysis_request : str = "Provide a comprehensive investment analysis",
):
    """
    Streams results as Server-Sent Events (SSE).

    WHY STREAMING:
        Full analysis takes 30-60 seconds. Without streaming the user
        sees nothing until it's all done. With SSE they see each agent
        completing in real time:
            "✓ Market data fetched"
            "✓ Risk analysis complete"
            "✓ Fundamental analysis complete"
            "✓ Report generated"

    SSE FORMAT:
        Each event is sent as:   data: {json}\n\n
        The double newline is required by the SSE protocol.
        The browser/client reads each event as it arrives.
    """
    ticker = ticker.upper().strip()
    logger.info(f"[API] GET /analyze/stream → ticker={ticker}")

    def generate():
        """Generator that yields SSE-formatted events one agent at a time."""
        try:
            for chunk in stream_financial_analysis(ticker, analysis_request):
                # chunk = {node_name: partial_state_update}
                for node_name, state_update in chunk.items():
                    event = {
                        "agent"  : node_name,
                        "status" : "completed",
                        # Exclude heavy/non-serializable fields from stream events
                        "data"   : {
                            k: v for k, v in state_update.items()
                            if k not in ["messages", "historical_prices"]
                        },
                    }
                    # SSE format: "data: {json}\n\n"
                    yield f"data: {json.dumps(event)}\n\n"

            # Final event signals the stream is complete
            yield f"data: {json.dumps({'agent': 'PIPELINE', 'status': 'complete'})}\n\n"

        except Exception as e:
            logger.error(f"[API] Streaming failed: {e}")
            yield f"data: {json.dumps({'agent': 'PIPELINE', 'status': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type = "text/event-stream",
        headers    = {
            "Cache-Control"    : "no-cache",   # never cache streaming responses
            "X-Accel-Buffering": "no",         # disable nginx buffering
        }
    )


# =============================================================================
# RAW METRICS ENDPOINT (no LLM — fast)
# =============================================================================

@app.get(
    "/metrics/{ticker}",
    tags    = ["Data"],
    summary = "Raw financial metrics without AI analysis (fast)",
)
async def get_raw_metrics(ticker: str):
    """
    Fetches raw financial data directly from yfinance — no LLM involved.

    Fast (2-5 seconds). Use this to:
        - Quickly verify yfinance is working
        - Build dashboards with raw data
        - Test tools without spending OpenAI credits
    """
    from tools.financial_tools import (
        fetch_stock_price, fetch_fundamental_data, calculate_risk_metrics
    )
    ticker = ticker.upper().strip()
    logger.info(f"[API] GET /metrics/{ticker}")

    try:
        return {
            "ticker"      : ticker,
            "timestamp"   : datetime.now().isoformat(),
            "price_data"  : fetch_stock_price.invoke({"ticker": ticker}),
            "risk_metrics": calculate_risk_metrics.invoke({"ticker": ticker}),
            "fundamentals": fetch_fundamental_data.invoke({"ticker": ticker}),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# STARTUP EVENT
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Runs once when the server starts. Good place to validate config."""
    logger.info(f"[STARTUP] Financial MAS API starting...")
    logger.info(f"[STARTUP] Model : {settings.llm_model}")
    logger.info(f"[STARTUP] API key set: {'YES' if settings.openai_api_key else 'NO — set OPENAI_API_KEY'}")
    logger.info(f"[STARTUP] Swagger UI → http://localhost:8000/docs")


# =============================================================================
# RUN DIRECTLY
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)