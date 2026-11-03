"""
agents/market_data_agent.py — Market Data Specialist Agent

First agent in the pipeline.
Fetches raw price data and company info using yfinance tools.

Pattern used: ReAct (Reason + Act)
    1. LLM reasons: "I need to call fetch_stock_price"
    2. LLM acts:    emits a tool_call
    3. We execute:  run the tool, get the result
    4. We observe:  feed result back to LLM as a ToolMessage
    5. LLM reasons: "now I need fetch_company_info"
    6. Repeat until LLM stops calling tools
"""

import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from core.config import settings, FinancialAnalysisState
from tools.financial_tools import fetch_stock_price, fetch_company_info, ALL_MARKET_TOOLS

logger = logging.getLogger(__name__)


# The system prompt is the agent's "job description"
# Clear, specific prompts are the single biggest factor in agent quality
SYSTEM_PROMPT = """You are a Market Data Agent at a quantitative hedge fund.

Your ONLY job is to fetch accurate market data for a given stock ticker.

STEPS:
1. Call fetch_stock_price(ticker) to get current price and OHLCV data
2. Call fetch_company_info(ticker) to get sector, industry, and description
3. Both tools must be called — do not skip either one

Do NOT analyse or interpret the data. Just fetch and return it.
If a tool fails, report the error but continue with the other tool.
Never make up or hallucinate any financial numbers.
"""


def market_data_agent_node(state: FinancialAnalysisState) -> dict:
    """
    Fetches stock price and company info using the ReAct tool-calling loop.

    Args:
        state: shared state — reads 'ticker' and 'analysis_request'

    Returns:
        partial state update with 'market_data' and 'company_name' filled in
    """
    ticker = state.get("ticker", "UNKNOWN").upper()
    logger.info(f"[MARKET DATA AGENT] Starting for {ticker}")

    # --- Initialise LLM ---
    # bind_tools() tells the LLM what tools it can call.
    # The LLM will emit tool_call messages when it wants to use a tool.
    llm = ChatOpenAI(
        model       = settings.llm_model,
        temperature = settings.analyst_temperature,
        api_key     = settings.openai_api_key,
    ).bind_tools(ALL_MARKET_TOOLS)

    # --- Build the opening messages ---
    # SystemMessage = persistent role instructions
    # HumanMessage  = the actual task for this run
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Fetch all market data for ticker: {ticker}\n"
                f"Context: {state.get('analysis_request', 'General analysis')}"
            )
        ),
    ]

    collected_data = {}   # accumulate tool results here
    max_iterations = 5    # safety limit — prevent infinite loops

    # --- ReAct loop ---
    # Each iteration: call LLM → check if it wants to call tools
    # → execute tools → feed results back → repeat
    # Loop ends when the LLM stops emitting tool_calls
    for i in range(max_iterations):
        logger.info(f"[MARKET DATA AGENT] LLM iteration {i + 1}")

        response = llm.invoke(messages)

        # If no tool_calls, the LLM is done — it gave its final text answer
        if not response.tool_calls:
            logger.info("[MARKET DATA AGENT] LLM finished — no more tool calls")
            break

        # Add the assistant's tool-call message to conversation history
        # This is required to keep the conversation coherent for the LLM
        messages.append(response)

        # Execute each tool the LLM requested
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]   # which tool
            tool_args = tool_call["args"]   # arguments the LLM chose

            logger.info(f"[MARKET DATA AGENT] Calling {tool_name}({tool_args})")

            if tool_name == "fetch_stock_price":
                tool_result = fetch_stock_price.invoke(tool_args)
                collected_data["price_data"] = tool_result

            elif tool_name == "fetch_company_info":
                tool_result = fetch_company_info.invoke(tool_args)
                collected_data["company_data"] = tool_result

            else:
                tool_result = {"error": f"Unknown tool: {tool_name}"}

            # Feed the result back to the LLM as a ToolMessage
            # tool_call_id links this result to the specific request above
            messages.append(
                ToolMessage(
                    content     = json.dumps(tool_result),
                    tool_call_id= tool_call["id"],
                )
            )

    # --- Package results ---
    price_data   = collected_data.get("price_data", {})
    company_data = collected_data.get("company_data", {})

    # Merge both dicts into one clean market_data block
    market_data  = {**price_data, **company_data}
    company_name = company_data.get("company_name", ticker)

    logger.info(f"[MARKET DATA AGENT] Done. company={company_name}, price=${price_data.get('current_price', 'N/A')}")

    # Return ONLY the fields we updated — LangGraph merges with existing state
    return {
        "market_data"  : market_data,
        "company_name" : company_name,
        "agent_steps"  : [
            f"MARKET_DATA_AGENT: price=${price_data.get('current_price', 'N/A')} | "
            f"company={company_name}"
        ],
    }