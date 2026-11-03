"""
core/config.py — System-wide Configuration & Shared State Schema
"""

import operator
from typing import Annotated, Any
from langgraph.graph import MessagesState
from pydantic_settings import BaseSettings
from pydantic import Field


# -----------------------------------------------------------------------------
# Settings — reads from your .env file automatically
# -----------------------------------------------------------------------------

class Settings(BaseSettings):
    """
    Loads config from environment variables / .env file.
    Priority: environment variable > .env file > default value
    """

    openai_api_key: str = Field(default="")

    # gpt-4o-mini is cheaper and fast — good for demos
    # swap to gpt-4o for better reasoning quality
    llm_model: str = Field(default="gpt-4o-mini")

    # Temperature = how creative/random the LLM is
    # 0.0 = fully deterministic (good for analysis)
    # 1.0 = creative (good for writing)
    supervisor_temperature: float = Field(default=0.0)
    analyst_temperature: float    = Field(default=0.1)
    writer_temperature: float     = Field(default=0.4)

    # App metadata
    app_name:    str = "Financial Multi-Agent System"
    app_version: str = "1.0.0"

    class Config:
        env_file = ".env"
        extra    = "ignore"


# Single shared instance — imported by every other module
settings = Settings()


# -----------------------------------------------------------------------------
# FinancialAnalysisState — the shared "baton" passed between all agents
# -----------------------------------------------------------------------------
# This is a TypedDict. Every agent receives the FULL state,
# does its work, and returns a PARTIAL update.
# LangGraph merges the update back into the state automatically.
#
# Annotated[list, operator.add] is a LangGraph "reducer":
#   it tells LangGraph to APPEND to the list instead of overwriting it.
#   Without this, Agent B would erase Agent A's work.

class FinancialAnalysisState(MessagesState):
    """
    Shared state dictionary — the single source of truth across all agents.

    Who fills what:
        market_data_agent       → market_data, company_name
        risk_analyst_agent      → risk_metrics, risk_insights
        fundamental_analyst     → fundamental_data, fundamental_insights
        report_writer_agent     → final_report, recommendation
        supervisor              → next_agent, analysis_complete
    """

    # --- Input (provided by user) ---
    ticker:           str          # e.g. "AAPL"
    company_name:     str          # e.g. "Apple Inc."
    analysis_request: str          # free-text instruction from user

    # --- Market Data Agent output ---
    market_data: dict[str, Any]    # price, volume, 52w range, company info

    # --- Risk Analyst Agent output ---
    risk_metrics: dict[str, Any]   # volatility, beta, sharpe, VaR
    risk_insights: Annotated[      # APPENDED across agents (not overwritten)
        list[str], operator.add
    ]

    # --- Fundamental Analyst Agent output ---
    fundamental_data: dict[str, Any]     # P/E, EPS, margins, debt
    fundamental_insights: Annotated[     # APPENDED across agents
        list[str], operator.add
    ]

    # --- Supervisor control ---
    next_agent:        str    # which agent to run next
    analysis_complete: bool   # True when report is done

    # --- Report Writer Agent output ---
    final_report:    str   # full Markdown investment brief
    recommendation:  str   # "BUY | Confidence: HIGH"

    # --- Audit trail ---
    agent_steps: Annotated[list[str], operator.add]  # log of every agent action
    errors:      Annotated[list[str], operator.add]  # non-fatal errors


# -----------------------------------------------------------------------------
# AgentNames — string constants to avoid typo bugs
# -----------------------------------------------------------------------------

class AgentNames:
    SUPERVISOR          = "supervisor"
    MARKET_DATA_AGENT   = "market_data_agent"
    RISK_ANALYST_AGENT  = "risk_analyst_agent"
    FUNDAMENTAL_ANALYST = "fundamental_analyst_agent"
    REPORT_WRITER       = "report_writer_agent"
    END                 = "END"